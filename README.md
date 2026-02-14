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



