"""Main entry point for the split LIETS-QSGS workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import SimulationConfig, build_neighbor_offsets
from export_raw_mhd import export_raw_mhd
from export_vtk import export_vtk
from growth_gpu import grow_phase_gpu
from postprocess import postprocess_geometry
from seeding_gpu import generate_seeds_gpu
from utils import print_system_mem
from visualization_xy import save_xy_slice_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the LIETS-QSGS workflow split out of the notebook.'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Directory for generated figures and export files. Default: current directory.',
    )
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display the XY slice figure in addition to saving it.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig()
    offsets = build_neighbor_offsets()

    print_system_mem('Before growth')
    seed_result = generate_seeds_gpu(config)
    growth_result = grow_phase_gpu(
        arrgrid=seed_result.arrgrid,
        solids=seed_result.solids,
        count=seed_result.num_seeds,
        offsets=offsets,
        target=seed_result.target_solids,
        config=config,
    )
    postprocess_result = postprocess_geometry(growth_result.arrgrid, config)

    xy_slice_path = output_dir / config.xy_slice_filename
    vtk_path = output_dir / config.vtk_filename
    raw_path = output_dir / config.raw_filename
    mhd_path = output_dir / config.mhd_filename

    save_xy_slice_comparison(
        original_grid=postprocess_result.original_grid,
        processed_grid=postprocess_result.connected_grid,
        output_path=xy_slice_path,
        show=args.show_plot,
    )

    # The original notebook exported the median-filtered grid, not the
    # connectivity-cleaned grid. That behavior is preserved here.
    export_vtk(postprocess_result.filtered_grid, config, output_path=vtk_path)
    export_raw_mhd(
        postprocess_result.filtered_grid,
        config,
        raw_path=raw_path,
        mhd_path=mhd_path,
    )

    print('Workflow complete.')
    print(f'XY slice figure: {xy_slice_path}')
    print(f'VTK file: {vtk_path}')
    print(f'RAW file: {raw_path}')
    print(f'MHD file: {mhd_path}')
    print(
        'Note: VTK and RAW/MHD exports use the median-filtered grid to '
        'match the original notebook export cells.'
    )


if __name__ == '__main__':
    main()
