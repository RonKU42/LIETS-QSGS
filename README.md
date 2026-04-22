# LIETS-QSGS

Ultra-Fast Granular Porous Media Generation: a GPU-Accelerated List-Indexed Explicit Time-Stepping QSGS Algorithm

## Authors

Ruofan Wang  
Department of Chemical and Petroleum Engineering, Khalifa University, Abu Dhabi, UAE  
Email: 100065182@ku.ac.ae

Mohammed Al Kobaisi  
Department of Chemical and Petroleum Engineering, Khalifa University, Abu Dhabi, UAE  
Email: mohammed.alkobaisi@ku.ac.ae

## Summary

LIETS-QSGS is a Python implementation of an accelerated Quartet Structure Generation Set (QSGS) workflow for granular porous media generation. The current implementation stores the three-dimensional phase field in a flattened one-dimensional array on the GPU and advances growth by explicit time stepping. A solid-voxel coordinate list is retained as an auxiliary structure for bookkeeping and reconstruction, but the growth update itself is performed on the flattened phase field.

The repository uses NumPy on CPU-side utilities and CuPy for GPU execution. It includes seed-spacing control through diamond dilation, a volume-fraction-dependent growth probability, postprocessing, and export utilities.

For a 400^3 domain, the LIETS implementation reported in the manuscript reduces generation time from tens of minutes for a serial CPU QSGS implementation and several minutes for vectorized CPU and GPU QSGS implementations to about 24 s on a consumer-grade RTX 4060, with peak throughputs up to 2.7 x 10^7 nodes/s. A Fontainebleau sandstone benchmark at 500^3 resolution shows that the generated structures reproduce the expected dependence of pore and grain size distributions on seed spacing and yield permeability-porosity trends within the reported experimental envelope.

## What is implemented in this repository

The current code path is consistent with the revised manuscript and uses the following growth logic:

1. The 3D binary phase field is flattened into a 1D GPU array.
2. At each global iteration, one random number is generated for each voxel in the domain.
3. A custom CUDA kernel assigns one thread to each voxel index.
4. For a void voxel, all 26 neighbor directions are checked.
5. The acceptance probability is computed from neighboring solid voxels through an effective growth probability.
6. Newly accepted voxels are stored in a temporary binary acceptance array.
7. If the number of accepted voxels exceeds the remaining target budget, a random subset is retained.
8. After growth, the phase field is reshaped back to the 3D grid and the solid-voxel coordinates are rebuilt.

In the present single-phase implementation, the same updated growth probability is used for all 26 neighbor directions.

## Current scope

This repository is intended for granular porous media generation, with validation centered on the Fontainebleau sandstone benchmark. It should be understood as an accelerated QSGS generator for sandstone-type granular porous media. Applicability to strongly heterogeneous unconventional rocks, such as shale and coal, has not yet been established in this work.

## Repository structure

- `QSGS_GPU_LIETS_spacing30_Size500_phi0.1500_1.ipynb` contains the original notebook.
- `config.py` stores default geometry, porosity, grain size, seed spacing, and output settings.
- `utils.py` contains helper utilities.
- `seeding_gpu.py` performs GPU seed generation with spacing control.
- `growth_gpu.py` contains the custom CUDA kernel, the growth routine, and rebuilding of solid coordinates.
- `postprocess.py` applies median filtering and pore connectivity cleanup.
- `visualization_xy.py` writes the XY slice comparison figure.
- `export_vtk.py` exports VTK output.
- `export_raw_mhd.py` exports RAW and MHD output.
- `main.py` runs the complete workflow.

## Requirements

Install the packages used by the workflow:

- Python 3.9 or newer
- NumPy
- SciPy
- Matplotlib
- psutil
- CuPy with CUDA support
- `cupyx.scipy.ndimage`

## How to run

Run the full workflow from the repository directory:

```bash
python main.py
```

Optional arguments:

```bash
python main.py --output-dir results
python main.py --show-plot
```

## Workflow executed by `main.py`

By default, `main.py` performs the following steps:

1. prints system memory usage;
2. generates seeds on the GPU with spacing control;
3. runs the LIETS-QSGS growth stage on the GPU;
4. applies median filtering and pore connectivity cleanup;
5. saves the XY slice comparison figure;
6. exports VTK output;
7. exports RAW and MHD output.

## Important implementation notes

- The manuscript timing and throughput values refer to the growth stage only unless otherwise stated.
- The VTK and RAW/MHD exports use the median-filtered grid, matching the original notebook export behavior.
- The XY slice comparison uses the connectivity-cleaned grid.
- The postprocessing stage is separate from the growth-stage benchmark timing.

## Parameters

The main parameters are stored in `SimulationConfig` in `config.py`.

Key parameters include:

- `lx`, `ly`, `lz`: domain size
- `phi`: porosity
- `voxel_size`: voxel size
- `grain_diameter`: target grain diameter
- `g_i_ref`: reference growth probability
- `s_sphere`: seed spacing parameter
- `n_rounds`: number of seed-generation rounds

## Limitations

Although the LIETS strategy substantially improves computational efficiency, the current GPU implementation still includes several full-domain auxiliary operations, such as random-field generation, voxel-state update arrays, and seed-spacing dilation. Performance therefore becomes increasingly memory-bound for very large samples, especially on consumer-grade GPUs with limited memory bandwidth and VRAM. The current validation is restricted to the Fontainebleau sandstone benchmark.

## Citation

If you use this repository, please cite the associated manuscript and the original Fast-QSGS work by Yang et al.

