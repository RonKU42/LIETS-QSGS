"""Notebook block for GPU growth after seed generation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cupy as cp
import numpy as np

from config import SimulationConfig
from utils import print_system_mem


grow_step_code = r'''
extern "C" __global__
void grow_step(const unsigned char* arr,
               unsigned char* new_flag,
               const int* offsets,
               const double* probs,
               const double* rand,
               int lx, int ly, int lz,
               int n_dirs)
{
    int total = lx * ly * lz;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    int yz  = ly * lz;
    int x   = idx / yz;
    int rem = idx % yz;
    int y   = rem / lz;
    int z   = rem % lz;

    if (arr[idx] != 0) {
        new_flag[idx] = 0;
        return;
    }

    bool has_solid = false;
    double p_prod = 1.0;

    for (int k = 0; k < n_dirs; ++k) {
        int dx = offsets[3 * k + 0];
        int dy = offsets[3 * k + 1];
        int dz = offsets[3 * k + 2];

        int nx = x - dx;
        int ny = y - dy;
        int nz = z - dz;

        if (nx < 0 || nx >= lx || ny < 0 || ny >= ly || nz < 0 || nz >= lz)
            continue;

        int nidx = nx * yz + ny * lz + nz;

        if (arr[nidx] == 1) {
            has_solid = true;
            double p = probs[k];
            p_prod *= (1.0 - p);
        }
    }

    if (!has_solid) {
        new_flag[idx] = 0;
        return;
    }

    double p_eff = 1.0 - p_prod;

    if (rand[idx] < p_eff) {
        new_flag[idx] = 1;
    } else {
        new_flag[idx] = 0;
    }
}
'''

grow_step_kernel = cp.RawKernel(grow_step_code, 'grow_step')


@dataclass
class GrowthResult:
    arrgrid: np.ndarray
    final_count: int
    elapsed: float
    computational_speed: float


def rebuild_solids_inplace(
    arrgrid: np.ndarray,
    solids: np.ndarray,
    lx: int,
    ly: int,
    lz: int,
) -> int:
    """Rebuild the solids coordinate array without a large temporary allocation."""
    max_solids = solids.shape[0]
    count = 0

    for x in range(lx):
        for y in range(ly):
            row = arrgrid[x, y, :]
            nz = np.nonzero(row)[0]
            k = nz.shape[0]
            if k == 0:
                continue

            if count + k > max_solids:
                k = max_solids - count
                if k <= 0:
                    return max_solids

            solids[count:count + k, 0] = x
            solids[count:count + k, 1] = y
            solids[count:count + k, 2] = nz[:k]
            count += k

            if count >= max_solids:
                return max_solids

    return count


def grow_phase_gpu(
    arrgrid: np.ndarray,
    solids: np.ndarray,
    count: int,
    offsets: np.ndarray,
    target: int,
    config: SimulationConfig,
) -> GrowthResult:
    """Run the LIETS-QSGS growth phase on the GPU."""
    print('Using GPU grow_phase_gpu')
    start_time = time.time()

    total_voxels = config.total_voxels
    arr_flat = cp.asarray(arrgrid.ravel(order='C'), dtype=cp.uint8)
    offsets_gpu = cp.asarray(offsets.reshape(-1), dtype=cp.int32)
    probs_gpu = cp.full(offsets.shape[0], config.g_i_ref, dtype=cp.float64)
    new_flag = cp.zeros(total_voxels, dtype=cp.uint8)

    n_dirs = offsets.shape[0]
    threads_per_block = 256
    blocks = (total_voxels + threads_per_block - 1) // threads_per_block

    count = int(cp.sum(arr_flat).get())

    while count < target:
        phi_a = float(count) / float(total_voxels)
        phi_s = config.phi_solid
        Gi = config.g_i_ref * (phi_s - 0.95 * phi_a) / (0.05 * phi_s)
        if Gi < 0.0:
            Gi = 0.0

        probs_gpu.fill(np.float64(Gi))
        rand_field = cp.random.random(total_voxels, dtype=cp.float64)
        new_flag.fill(0)

        grow_step_kernel(
            (blocks,),
            (threads_per_block,),
            (
                arr_flat,
                new_flag,
                offsets_gpu,
                probs_gpu,
                rand_field,
                config.lx,
                config.ly,
                config.lz,
                n_dirs,
            ),
        )

        new_count = int(cp.count_nonzero(new_flag).get())
        if new_count == 0:
            break

        remaining = target - count

        if new_count <= remaining:
            arr_flat[new_flag == 1] = 1
            count += new_count
        else:
            idxs = cp.where(new_flag == 1)[0]
            perm = cp.random.permutation(idxs.shape[0])
            keep = idxs[perm[:remaining]]
            arr_flat[keep] = 1
            count += remaining
            break

    cp.cuda.Stream.null.synchronize()
    print_system_mem('After growth')

    arrgrid[:, :, :] = cp.asnumpy(arr_flat.reshape(config.lx, config.ly, config.lz))
    final_count = rebuild_solids_inplace(arrgrid, solids, config.lx, config.ly, config.lz)

    elapsed = time.time() - start_time
    computational_speed = config.total_voxels / elapsed if elapsed > 0.0 else float('inf')

    print(f'Growth complete: total solids = {final_count}')
    print(f'Elapsed time (growth): {elapsed:.2f} s')
    print(f'Computational Speed: {computational_speed:.2f} node/s')

    return GrowthResult(
        arrgrid=arrgrid,
        final_count=final_count,
        elapsed=elapsed,
        computational_speed=computational_speed,
    )
