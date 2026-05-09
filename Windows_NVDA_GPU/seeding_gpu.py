"""Notebook block for GPU seed generation and spacing control."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import binary_dilation

from config import SimulationConfig


@dataclass
class SeedResult:
    arrgrid: np.ndarray
    solids: np.ndarray
    num_seeds: int
    target_solids: int
    target_seeds: int
    elapsed: float


def generate_seeds_gpu(config: SimulationConfig) -> SeedResult:
    """Generate spaced seeds on the GPU.

    This function is a script-friendly version of notebook cell 1.
    """
    print(f"Computed seed volume fraction Sd = {config.seed_prob:.3e}")
    print(
        'Using equal-volume octahedral spacing radius: '
        f's_sphere = {config.s_sphere}, s_octa = {config.s_octa}'
    )
    print(f'Target solid voxels: {config.target_solids}')

    target_seeds = int(config.seed_prob * config.total_voxels)
    print(f'Target seed count: {target_seeds}')

    start_time = time.time()

    diamond_struct = cp.zeros((3, 3, 3), dtype=cp.bool_)
    diamond_struct[1, 1, 0] = True
    diamond_struct[1, 1, 2] = True
    diamond_struct[1, 0, 1] = True
    diamond_struct[1, 2, 1] = True
    diamond_struct[0, 1, 1] = True
    diamond_struct[2, 1, 1] = True

    accepted = cp.zeros((config.lx, config.ly, config.lz), dtype=cp.bool_)
    forbidden = cp.zeros((config.lx, config.ly, config.lz), dtype=cp.bool_)

    for it in range(config.n_rounds):
        seeds_before = int(cp.count_nonzero(accepted).get())
        remaining_seeds = target_seeds - seeds_before
        if remaining_seeds <= 0:
            print(f'Reached target seeds at round {it}, current seeds: {seeds_before}')
            break

        candidate_mask_base = (~forbidden) & (~accepted)
        free_count = int(cp.count_nonzero(candidate_mask_base).get())
        if free_count == 0:
            print(f'No free voxels left at round {it}, stop.')
            break

        target_this_round = max(1, int(config.alpha * remaining_seeds))
        p_raw = config.oversample_factor * target_this_round / free_count
        p_raw = min(p_raw, 1.0)

        rnd = cp.random.random((config.lx, config.ly, config.lz))
        candidate = candidate_mask_base & (rnd < p_raw)
        num_candidates = int(cp.count_nonzero(candidate).get())

        print(
            f'Round {it + 1}, free voxels: {free_count}, '
            f'p_raw: {p_raw:.3e}, candidates: {num_candidates}'
        )

        if num_candidates == 0:
            continue

        accepted |= candidate
        forbidden = binary_dilation(
            accepted,
            structure=diamond_struct,
            iterations=config.s_octa,
            border_value=0,
        )

        seeds_after = int(cp.count_nonzero(accepted).get())
        print(
            f'End of round {it + 1}, new seeds: {seeds_after - seeds_before}, '
            f'total seeds: {seeds_after}'
        )

        if seeds_after >= target_seeds:
            print(
                f'Reached target seeds at round {it + 1} during update, '
                f'current seeds: {seeds_after}'
            )
            break

    accepted_np = accepted.get()
    arrgrid = np.zeros((config.lx, config.ly, config.lz), dtype=np.uint8)
    arrgrid[accepted_np] = 1

    seeds = np.argwhere(accepted_np)
    num_seeds = seeds.shape[0]
    print(f'Final seeds after spacing: {num_seeds}')
    if num_seeds == 0:
        raise RuntimeError(
            "No seeds generated after spacing. Adjust 'seed_prob_override' or 's_sphere'."
        )

    solids = np.zeros((config.target_solids, 3), dtype=np.int32)
    solids[:num_seeds, :] = seeds

    elapsed = time.time() - start_time
    print(f'Elapsed time (seeding): {elapsed:.2f} s')

    return SeedResult(
        arrgrid=arrgrid,
        solids=solids,
        num_seeds=num_seeds,
        target_solids=config.target_solids,
        target_seeds=target_seeds,
        elapsed=elapsed,
    )
