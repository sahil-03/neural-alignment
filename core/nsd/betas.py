from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import sparse

from core.datasets.arrays import ImageBetas, TrialBetas
from core.nsd.constants import BETA_SCALE, NSD_SESSIONS, TRIALS_PER_SESSION
from core.nsd.meta import load_trials
from core.nsd.remote import NSDBucket, NSDKeys, nsd_bucket
from core.nsd.voxels import VoxelMask
from core.storage import backends
from core.storage.store import Store


class SubjectBetaSummary(BaseModel):
    subject: int
    n_images: int
    n_voxels: int
    reps_hist: list[int]        # reps_hist[k] = number of images seen k times
    repeat_r_mean: float        # mean per-voxel test-retest correlation
    repeat_r_p90: float


class OrientationReport(BaseModel):
    subject: int
    session: int
    n_voxels: int
    corr: float
    exact: bool


def zscore_columns(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(0)) / np.maximum(x.std(0), eps)


def average_repeats(ids: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean response per unique id -> (sorted unique ids, means float32, counts)."""
    codes, uniq = pd.factorize(ids, sort=True)
    n = len(ids)
    A = sparse.csr_matrix((np.ones(n, np.float32), (codes, np.arange(n))), shape=(len(uniq), n))
    counts = np.bincount(codes, minlength=len(uniq))
    means = np.asarray(A @ Y.astype(np.float32)) / counts[:, None]
    return np.asarray(uniq, dtype=np.int64), means.astype(np.float32), counts


def repeat_reliability(ids: np.ndarray, trial: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson r between the 1st and 2nd presentation of each repeated image. -> (V,)

    Clearly > 0 in visual cortex if trials are correctly matched to images; ~0 if misaligned.
    """
    order = np.lexsort((trial, ids))
    sorted_ids = ids[order]
    is_first = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
    group_start = np.maximum.accumulate(np.where(is_first, np.arange(len(order)), 0))
    rank = np.arange(len(order)) - group_start
    second = order[rank == 1]
    first = order[(rank == 0) & np.isin(sorted_ids, ids[second])]
    if len(first) < 2:
        raise ValueError("need at least two repeated images to estimate reliability")
    a = zscore_columns(Y[first].astype(np.float32))
    b = zscore_columns(Y[second].astype(np.float32))
    return (a * b).mean(0)


def load_session(
    subject: int, session: int, mask: VoxelMask, workdir: Path, bucket: NSDBucket | None = None
) -> np.ndarray:
    """Download one session (~1 GB), mask, rescale, z-score each voxel within the session -> (750, V) float32.

    The HDF5 is chunked one-voxel-by-all-trials, so partial remote reads are impractical; we
    download the whole file and delete it right after.
    """
    local = (bucket or nsd_bucket()).download(NSDKeys.betas_hdf5(subject, session), workdir / "betas.hdf5")
    try:
        with h5py.File(local, "r") as h:
            raw = h["betas"][:]  # (750, Z, Y, X) int16
    finally:
        local.unlink(missing_ok=True)
    if raw.shape[0] != TRIALS_PER_SESSION:
        raise ValueError(f"subj{subject:02d} session {session}: expected {TRIALS_PER_SESSION} trials, got {raw.shape[0]}")
    x = mask.from_hdf5(raw).astype(np.float32) / BETA_SCALE
    return zscore_columns(x)


def process_session(
    store: Store, subject: int, session: int, *, force: bool = False,
    mask: VoxelMask | None = None, bucket: NSDBucket | None = None,
) -> bool:
    """Write one session's masked, z-scored betas (float16). Returns False if it already existed."""
    if not 1 <= session <= NSD_SESSIONS[subject]:
        raise ValueError(f"subj{subject:02d} has sessions 1..{NSD_SESSIONS[subject]}, got {session}")
    key = backends.session_betas(subject, session)
    if not force and store.exists(key):
        return False
    mask = mask or VoxelMask.nsdgeneral(subject, bucket)
    with tempfile.TemporaryDirectory() as d:
        y = load_session(subject, session, mask, Path(d), bucket)
    store.save_array(key, y.astype(np.float16))
    return True


def missing_sessions(store: Store, subject: int) -> list[int]:
    return [k for k in range(1, NSD_SESSIONS[subject] + 1) if not store.exists(backends.session_betas(subject, k))]


def aggregate_subject(
    store: Store, subject: int, *, force: bool = False, bucket: NSDBucket | None = None
) -> SubjectBetaSummary:
    """Stack all sessions into TrialBetas, average repeats into ImageBetas, and summarize."""
    if force or not (TrialBetas.exists(store, backends.trial_betas(subject))
                     and ImageBetas.exists(store, backends.image_betas(subject))):
        missing = missing_sessions(store, subject)
        if missing:
            raise FileNotFoundError(f"subj{subject:02d}: sessions not processed yet: {missing}")

        trials = load_trials(store, subject)
        Y = np.concatenate(
            [store.load_array(backends.session_betas(subject, k)) for k in range(1, NSD_SESSIONS[subject] + 1)]
        )
        # Session files hold 750 rows in within-session order; the trial table is sorted by trial number.
        if len(Y) != len(trials):
            raise ValueError(f"{len(Y)} beta rows vs {len(trials)} trials")
        tb = TrialBetas(trial=trials["trial"].to_numpy(), ids=trials["coco_id"].to_numpy(), Y=Y)
        tb.save(store, backends.trial_betas(subject))

        mask = VoxelMask.nsdgeneral(subject, bucket)
        ncsnr = mask.from_nifti((bucket or nsd_bucket()).read_nifti(NSDKeys.ncsnr(subject)))
        ids, means, counts = average_repeats(tb.ids, tb.Y)
        ImageBetas(
            ids=ids, Y=means, n_reps=counts, ncsnr=ncsnr.astype(np.float32), voxel_ijk=mask.ijk,
        ).save(store, backends.image_betas(subject))

    return summarize_subject(store, subject)


def summarize_subject(store: Store, subject: int) -> SubjectBetaSummary:
    tb = TrialBetas.load(store, backends.trial_betas(subject))
    ib = ImageBetas.load(store, backends.image_betas(subject))
    rel = repeat_reliability(tb.ids, tb.trial, tb.Y)
    return SubjectBetaSummary(
        subject=subject,
        n_images=len(ib),
        n_voxels=ib.n_voxels,
        reps_hist=np.bincount(ib.n_reps).tolist(),
        repeat_r_mean=float(rel.mean()),
        repeat_r_p90=float(np.percentile(rel, 90)),
    )


def verify_orientation(
    subject: int = 1, session: int = 1, n_trials: int = 3, bucket: NSDBucket | None = None
) -> OrientationReport:
    """Compare the HDF5 and NIfTI copies of the same session through VoxelMask; they must agree.

    Downloads ~1.5 GB, so run it remotely.
    """
    bucket = bucket or nsd_bucket()
    mask = VoxelMask.nsdgeneral(subject, bucket)
    with tempfile.TemporaryDirectory() as d:
        h5 = bucket.download(NSDKeys.betas_hdf5(subject, session), Path(d) / "b.hdf5")
        nii = bucket.download(NSDKeys.betas_nifti(subject, session), Path(d) / "b.nii.gz")
        with h5py.File(h5, "r") as h:
            a = mask.from_hdf5(h["betas"][:n_trials]).astype(np.float64)
        b = mask.from_nifti(np.asarray(nib.Nifti1Image.from_filename(nii).dataobj[..., :n_trials], dtype=np.float64))
    report = OrientationReport(
        subject=subject, session=session, n_voxels=mask.n_voxels,
        corr=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), exact=bool(np.array_equal(a, b)),
    )
    if report.corr < 0.999:
        raise AssertionError(f"HDF5/NIfTI orientation mismatch: {report}")
    return report
