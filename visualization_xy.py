"""Notebook block for the mid-plane XY slice visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def save_xy_slice_comparison(
    original_grid: np.ndarray,
    processed_grid: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Save or show the notebook's XY slice comparison figure."""
    z_mid = processed_grid.shape[2] // 2
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow((original_grid == 0)[:, :, z_mid].T, origin='lower', cmap='gray')
    axs[0].set_title('Original Pores XY Slice')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    axs[1].imshow((processed_grid == 0)[:, :, z_mid].T, origin='lower', cmap='gray')
    axs[1].set_title('Processed Pores XY Slice')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    plt.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches='tight')
        print(f'XY slice figure written to {saved_path}')

    if show:
        plt.show()

    plt.close(fig)
    return saved_path
