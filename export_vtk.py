"""Notebook block for VTK export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from config import SimulationConfig


def export_vtk(
    arrgrid: np.ndarray,
    config: SimulationConfig,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the grid to a VTK legacy file for ParaView."""
    vtk_path = Path(output_path or config.vtk_filename)
    vtk_path.parent.mkdir(parents=True, exist_ok=True)

    flat_data = arrgrid.flatten(order='C')
    with vtk_path.open('w', encoding='utf-8') as vtk_file:
        vtk_file.write(
            '# vtk DataFile Version 3.0\n'
            'QSGS 3D model output\n'
            'ASCII\n'
            'DATASET STRUCTURED_POINTS\n'
            f'DIMENSIONS {config.lx} {config.ly} {config.lz}\n'
            'ORIGIN 0 0 0\n'
            f'SPACING {config.voxel_size:g} {config.voxel_size:g} {config.voxel_size:g}\n'
            f'POINT_DATA {config.total_voxels}\n'
            'SCALARS phase unsigned_char 1\n'
            'LOOKUP_TABLE default\n'
        )
        for value in flat_data:
            vtk_file.write(f'{int(value)}\n')

    print(f'VTK file written to {vtk_path}')
    return vtk_path
