# LIETS-QSGS

Author information:

Ruofan Wang

100065182@ku.ac.ae

Department of Chemical and Petroleum Engineering
Khalifa University, Abu Dhabi, UAE


Mohammed Al Kobaisi*

mohammed.alkobaisi@ku.ac.ae

Department of Chemical and Petroleum Engineering
Khalifa University, Abu Dhabi, UAE

Article information:

Ultra-Fast 3D Porous Media Generation: a GPU- Accelerated List-Indexed Explicit Time-Stepping QSGS Algorithm

https://arxiv.org/abs/2602.11734

Abstract: Efficient generation of high-resolution synthetic microstructures is essential in digital rock physics, yet classical Quartet Structure Generation Set (QSGS) algorithms become prohibitively expensive on large three-dimensional grids. We develop a list-indexed explicit time-stepping (LIETS) formulation of QSGS that restricts stochastic growth operations to an explicit active front instead of the entire voxel grid. The method is implemented in Python using NumPy on CPUs and CuPy on GPUs, and incorporates seed-spacing control via diamond dilation together with a volume-fraction-dependent directional growth probability. For a 400^3 domain, LIETS reduces generation time from tens of minutes for a serial CPU implementation and several minutes for vectorized CPU and GPU QSGS to about 24 s on a consumer-grade RTX 4060, achieving peak throughputs up to 2.7x10^7 nodes/s. A Fontainebleau sandstone benchmark at 500^3 resolution shows that LIETS reproduces the dependence of pore and grain size distributions on seed spacing (optimal s=30 voxels) and yields permeability-porosity trends within the experimental envelope and consistent with previously published Fast-QSGS results.

## Notebook-to-script mapping

The notebook was split according to its code blocks:

- **Cell 0** import statements are distributed across the modules that need them.
- **Cell 1** becomes `config.py`, `utils.py`, and `seeding_gpu.py`.
- **Cell 2** becomes `growth_gpu.py`.
- **Cell 3** becomes `postprocess.py`.
- **Cell 4** becomes `visualization_xy.py`.
- **Cell 5** becomes `export_vtk.py`.
- **Cell 6** becomes `export_raw_mhd.py`.
- `main.py` is the new top-level entry point that runs the full workflow in order.

## Files

- `config.py` centralizes the default domain size, porosity, voxel size, grain diameter, seed spacing, and output filenames.
- `utils.py` contains the `print_system_mem()` helper from the notebook.
- `seeding_gpu.py` runs the GPU seed-spacing stage and returns the initial `arrgrid` and `solids` arrays.
- `growth_gpu.py` contains the custom CUDA kernel, the solid-list rebuild helper, and the GPU growth routine.
- `postprocess.py` applies the median filter, performs connectivity analysis, and removes isolated pore regions.
- `visualization_xy.py` saves the mid-plane XY comparison figure.
- `export_vtk.py` writes the VTK legacy file for ParaView.
- `export_raw_mhd.py` writes the RAW and MHD volume files.
- `main.py` orchestrates all steps and serves as the recommended way to run the project.

## Requirements

Install the packages used in the notebook:

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- psutil
- CuPy with CUDA support
- `cupyx.scipy.ndimage` via CuPy

## How to run

Run the full workflow from the directory that contains these files:

```bash
python main.py
```

Optional arguments:

```bash
python main.py --output-dir results
python main.py --show-plot
```

By default, `main.py` does the following:

1. prints system memory usage,
2. generates GPU seeds with spacing control,
3. runs the LIETS-QSGS growth stage on the GPU,
4. applies median filtering and pore connectivity cleanup,
5. saves the XY slice comparison figure as `core_xy_slice.png`,
6. exports `500_Spacing30_phi01500.vtk`, and
7. exports `500_Spacing30_phi01500.raw` plus `500_Spacing30_phi01500.mhd`.

## Parameter changes

The notebook's default parameters are stored in `SimulationConfig` inside `config.py`.
To change the synthetic sample settings, edit:

- `lx`, `ly`, `lz`
- `phi`
- `voxel_size`
- `grain_diameter`
- `g_i_ref`
- `s_sphere`
- `n_rounds`

## Behavior preserved from the notebook

- The default numerical values are copied from the uploaded notebook.
- The VTK and RAW/MHD export steps write the **median-filtered** grid, because that is what the original notebook export cells used.
- The connectivity-cleaned grid is still available inside `postprocess.py` and is used for the XY slice comparison plot.


