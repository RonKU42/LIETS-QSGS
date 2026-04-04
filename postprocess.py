"""Notebook block for median filtering and pore connectivity cleanup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, median_filter

from config import SimulationConfig


@dataclass
class PostprocessResult:
    original_grid: np.ndarray
    filtered_grid: np.ndarray
    connected_grid: np.ndarray
    porosity_after_filtering: float
    connected_porosity_ratio: float
    num_pore_regions: int


def postprocess_geometry(
    arrgrid: np.ndarray,
    config: SimulationConfig,
) -> PostprocessResult:
    """Apply the notebook's median filtering and connectivity analysis."""
    arrgrid_original = arrgrid.copy()
    filtered_grid = median_filter(arrgrid, size=3)

    total_solid = int(filtered_grid.sum())
    porosity = 1.0 - total_solid / config.total_voxels
    print(f'Porosity after filtering: {porosity:.4f}')

    connected_grid = filtered_grid.copy()
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, num_features = label(connected_grid == 0, structure=structure)
    print(f'Detected {num_features} pore regions before removal.')

    inlet_labels = np.unique(labels[:, :, 0][labels[:, :, 0] > 0])
    outlet_labels = np.unique(labels[:, :, config.lz - 1][labels[:, :, config.lz - 1] > 0])
    good_labels = np.intersect1d(inlet_labels, outlet_labels)

    mask_remove = (labels > 0) & (~np.isin(labels, good_labels))
    connected_grid[mask_remove] = 1

    connected_ratio = np.count_nonzero(connected_grid == 0) / config.total_voxels
    print(f'Connected porosity ratio: {connected_ratio:.4f}')

    return PostprocessResult(
        original_grid=arrgrid_original,
        filtered_grid=filtered_grid,
        connected_grid=connected_grid,
        porosity_after_filtering=porosity,
        connected_porosity_ratio=connected_ratio,
        num_pore_regions=int(num_features),
    )
