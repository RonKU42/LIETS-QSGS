"""Central configuration for the script version of the LIETS-QSGS notebook."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def build_neighbor_offsets() -> np.ndarray:
    """Return the 26-neighbor offsets used by the QSGS growth step."""
    offsets = np.array(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ],
        dtype=np.int32,
    )
    assert offsets.shape[0] == 26
    return offsets


@dataclass(frozen=True)
class SimulationConfig:
    """Default parameters copied from the uploaded notebook."""

    lx: int = 500
    ly: int = 500
    lz: int = 500
    phi: float = 0.1625
    voxel_size: float = 4.0
    grain_diameter: float = 160.0
    g_i_ref: float = 8e-4
    s_sphere: int = 30
    n_rounds: int = 50
    alpha: float = 0.2
    oversample_factor: float = 2.0
    seed_prob_override: Optional[float] = None
    vtk_filename: str = '500_Spacing30_phi01500.vtk'
    raw_filename: str = '500_Spacing30_phi01500.raw'
    mhd_filename: str = '500_Spacing30_phi01500.mhd'
    xy_slice_filename: str = 'core_xy_slice.png'

    @property
    def total_voxels(self) -> int:
        return self.lx * self.ly * self.lz

    @property
    def phi_solid(self) -> float:
        return 1.0 - self.phi

    @property
    def target_solids(self) -> int:
        return int((1.0 - self.phi) * self.total_voxels)

    @property
    def seed_prob(self) -> float:
        if self.seed_prob_override is not None:
            return self.seed_prob_override
        return (
            6.0
            * self.phi_solid
            * self.voxel_size ** 3
            / (np.pi * self.grain_diameter ** 3)
        )

    @property
    def s_octa(self) -> int:
        return int(round((math.pi ** (1.0 / 3.0)) * self.s_sphere))
