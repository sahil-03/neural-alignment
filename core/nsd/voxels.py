from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from core.nsd.remote import NSDBucket, NSDKeys, nsd_bucket


class VoxelMask(BaseModel):
    """Boolean ROI on the func1pt8mm grid, stored in NIfTI (X, Y, Z) orientation.

    NSD's HDF5 betas are (trial, Z, Y, X), the axis-reversed NIfTI layout. Both accessors return
    voxels in the same order (C-order over NIfTI X, Y, Z), so arrays from either source line up
    column-for-column. `core.nsd.betas.verify_orientation` checks this against real data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mask: np.ndarray  # (X, Y, Z) bool

    @field_validator("mask", mode="after")
    @classmethod
    def _check_mask(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 3 or v.dtype != bool:
            raise ValueError(f"mask must be a 3-D bool array, got {v.dtype} {v.shape}")
        if not v.any():
            raise ValueError("mask selects no voxels")
        return v

    @classmethod
    def nsdgeneral(cls, subject: int, bucket: NSDBucket | None = None) -> VoxelMask:
        vol = (bucket or nsd_bucket()).read_nifti(NSDKeys.nsdgeneral(subject))
        return cls(mask=vol == 1)  # 1 = in ROI, 0 = not, -1 = outside brain

    @property
    def n_voxels(self) -> int:
        return int(self.mask.sum())

    @property
    def ijk(self) -> np.ndarray:
        """(V, 3) NIfTI voxel coordinates, in the same order as the accessors below."""
        return np.argwhere(self.mask)

    def from_nifti(self, vol: np.ndarray) -> np.ndarray:
        """(X, Y, Z) -> (V,)   or   (X, Y, Z, T) -> (T, V)."""
        if vol.shape[:3] != self.mask.shape:
            raise ValueError(f"volume {vol.shape} does not match mask {self.mask.shape}")
        out = vol[self.mask]
        return out.T if out.ndim == 2 else out

    def from_hdf5(self, betas: np.ndarray) -> np.ndarray:
        """(T, Z, Y, X) -> (T, V)."""
        if betas.shape[1:] != self.mask.shape[::-1]:
            raise ValueError(f"betas {betas.shape} do not match reversed mask {self.mask.shape[::-1]}")
        return betas.transpose(0, 3, 2, 1)[:, self.mask]
