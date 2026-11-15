"""Anonymous, read-only access to the public NSD bucket (s3://natural-scenes-dataset, us-east-2)."""

from __future__ import annotations

import gzip
import io
from functools import cache
from pathlib import Path

import boto3
import nibabel as nib
import numpy as np
import pandas as pd
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from core.nsd.constants import NSD_BUCKET, NSD_REGION

_MB = 1024**2


class NSDKeys:
    """Object keys inside the NSD bucket."""

    STIM_INFO = "nsddata/experiments/nsd/nsd_stim_info_merged.csv"

    @staticmethod
    def nsdgeneral(subject: int) -> str:
        return f"nsddata/ppdata/subj{subject:02d}/func1pt8mm/roi/nsdgeneral.nii.gz"

    @staticmethod
    def _betas_dir(subject: int) -> str:
        # "GLMdenoise_RR" is NSD's directory name for the GLMsingle beta version (b3).
        return f"nsddata_betas/ppdata/subj{subject:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR"

    @classmethod
    def betas_hdf5(cls, subject: int, session: int) -> str:
        return f"{cls._betas_dir(subject)}/betas_session{session:02d}.hdf5"

    @classmethod
    def betas_nifti(cls, subject: int, session: int) -> str:
        return f"{cls._betas_dir(subject)}/betas_session{session:02d}.nii.gz"

    @classmethod
    def ncsnr(cls, subject: int) -> str:
        return f"{cls._betas_dir(subject)}/ncsnr.nii.gz"


class NSDBucket:
    def __init__(self, max_concurrency: int = 16):
        self._s3 = boto3.client(
            "s3",
            region_name=NSD_REGION,
            config=Config(
                signature_version=UNSIGNED,
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max_concurrency,
            ),
        )
        self._transfer = TransferConfig(multipart_chunksize=32 * _MB, max_concurrency=max_concurrency)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self._s3.get_object(Bucket=NSD_BUCKET, Key=key)["Body"].read()
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                raise FileNotFoundError(f"s3://{NSD_BUCKET}/{key}") from err
            raise

    def download(self, key: str, dest: str | Path) -> Path:
        """Parallel multipart download (used for the ~1 GB beta files)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(NSD_BUCKET, key, str(dest), Config=self._transfer)
        return dest

    def read_csv(self, key: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.get_bytes(key)), **kwargs)

    def read_nifti(self, key: str) -> np.ndarray:
        """For small volumes (masks, per-voxel maps). Returns data in NIfTI (X, Y, Z[, T]) orientation."""
        raw = self.get_bytes(key)
        img = nib.Nifti1Image.from_bytes(gzip.decompress(raw) if key.endswith(".gz") else raw)
        return np.asarray(img.get_fdata())


@cache
def nsd_bucket() -> NSDBucket:
    return NSDBucket()
