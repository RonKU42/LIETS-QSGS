"""Notebook block for RAW and MHD export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from config import SimulationConfig


def export_raw_mhd(
    arrgrid: np.ndarray,
    config: SimulationConfig,
    raw_path: Optional[Union[str, Path]] = None,
    mhd_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Write the grid to RAW and MHD files."""
    raw_path = Path(raw_path or config.raw_filename)
    mhd_path = Path(mhd_path or config.mhd_filename)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mhd_path.parent.mkdir(parents=True, exist_ok=True)

    arrgrid.astype(np.uint8).tofile(raw_path)
    spacing = (config.voxel_size, config.voxel_size, config.voxel_size)
    offset = (0, 0, 0)
    unit = 1e-6

    raw_reference = raw_path.name if raw_path.parent == mhd_path.parent else raw_path.as_posix()

    with mhd_path.open('w', encoding='utf-8') as mhd:
        mhd.write(
            'ObjectType = Image\n'
            'NDims = 3\n'
            'ElementType = MET_UCHAR\n'
            '\n'
            f'DimSize =        {config.lx}  {config.ly}  {config.lz}\n'
            f'ElementSpacing = {spacing[0]}   {spacing[1]}   {spacing[2]}\n'
            f'Offset =         {offset[0]}   {offset[1]}   {offset[2]}\n'
            f'Unit = {unit}\n'
            f'ElementDataFile = {raw_reference}\n'
        )

    print(f'RAW file written to {raw_path}')
    print(f'MHD header written to {mhd_path}')
    return raw_path, mhd_path
