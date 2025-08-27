# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.experimental import concat_where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import _calculate_nabla2_for_w
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_diffusion_to_w(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    diff_multfac_w: wpfloat,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """Core stencil consisting only of lines 58-64 from the original implementation."""
    z_nabla2_c = _calculate_nabla2_for_w(w_old, geofac_n2s)
    return concat_where(
        (interior_idx <= CellDim) & (CellDim < halo_idx),
        _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w),
        w_old,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def apply_diffusion_to_w(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    diff_multfac_w: wpfloat,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
    num_cells: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_diffusion_to_w(
        area,
        geofac_n2s,
        w_old,
        diff_multfac_w,
        interior_idx,
        halo_idx,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


if __name__ == "__main__":
    import argparse
    import time
    import numpy as np
    import cupy as cp

    from mklein_test import (
        get_torus_grid,
        trim_grid,
        reindex_cells,
        reindex_edges,
        reindex_vertices,
        sort_into_blocks_32,
        ToZeroBasedIndexTransformation,
    )
    from icon4py.model.common import dimension as dims
    from icon4py.model.common.dimension import CellDim, KDim
    from gt4py.next import int32

    # Select backend (CPU/GPU)
    b_end = gtx.gtfn_gpu
    xp = cp if "gpu" in str(b_end).lower() else np

    parser = argparse.ArgumentParser(
        description="Run apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence with optional block sorting",
    )
    parser.add_argument(
        "--block-sort",
        action="store_true",
        help="Sort edges and cells into blocks of 32 (same as *_on_32 scripts)",
    )
    args = parser.parse_args()

    grid_file = "../all_torus_files/torus_100000_100000_256_reordered.nc"
    levels = 80

    load_start = time.perf_counter()
    grid = get_torus_grid(grid_file, levels, ToZeroBasedIndexTransformation())
    print(f"Grid loaded in {time.perf_counter() - load_start:.6f}s")

    vertices, edges, cells = trim_grid(grid_file, grid)

    if args.block_sort:
        # Optional: use the block-sorting variant that packs indices into 32-element blocks
        edges, cells = sort_into_blocks_32(grid, grid_file, edges, cells)
        reindex_vertices(grid, vertices)
    else:
        # Original re-indexing logic
        reindex_cells(grid, cells)
        reindex_edges(grid, edges)
        reindex_vertices(grid, vertices)

    # Make sure connectivities live on the correct backend memory
    if xp.__name__ == "cupy":
        cp.get_default_memory_pool().free_all_blocks()
        for name, connectivity in grid.connectivities.items():
            if hasattr(connectivity, "ndarray"):
                connectivity.ndarray = cp.asarray(connectivity.ndarray)
            else:
                grid.connectivities[name] = cp.asarray(connectivity)

        # Convert offset providers for GPU
        for name, provider in grid.offset_providers.items():
            if hasattr(provider, "ndarray"):
                grid.offset_providers[name] = gtx.as_connectivity(
                    provider.domain,
                    codomain=provider.codomain,
                    data=provider.ndarray,
                    skip_value=None,
                    allocator=b_end,
                )

    rng = np.random.default_rng(1)

    # ----- Allocate and wrap input / output fields -----
    area_arr = xp.asarray(rng.random((grid.num_cells,)))
    c2e2co_len = grid.get_offset_provider("C2E2CO").ndarray.shape[1]
    geofac_n2s_arr = xp.asarray(rng.random((grid.num_cells, c2e2co_len)))
    geofac_grg_x_arr = xp.asarray(rng.random((grid.num_cells, c2e2co_len)))
    geofac_grg_y_arr = xp.asarray(rng.random((grid.num_cells, c2e2co_len)))
    w_old_arr = xp.asarray(rng.random((grid.num_cells, levels)))
    diff_multfac_n2w_arr = xp.asarray(rng.random((levels,)))

    # Create domains for field allocation
    cell_k_domain = gtx.domain({CellDim: grid.num_cells, KDim: levels})
    k_domain = gtx.domain({KDim: levels})
    cell_domain = gtx.domain({CellDim: grid.num_cells})
    geofac_domain = gtx.domain({CellDim: grid.num_cells, dims.C2E2CODim: c2e2co_len})

    area = gtx.as_field(cell_domain, area_arr, allocator=b_end)
    geofac_n2s = gtx.as_field(geofac_domain, geofac_n2s_arr, allocator=b_end)
    geofac_grg_x = gtx.as_field(geofac_domain, geofac_grg_x_arr, allocator=b_end)
    geofac_grg_y = gtx.as_field(geofac_domain, geofac_grg_y_arr, allocator=b_end)
    w_old = gtx.as_field(cell_k_domain, w_old_arr, allocator=b_end)
    w = gtx.zeros(cell_k_domain, allocator=b_end)
    dwdx = gtx.zeros(cell_k_domain, allocator=b_end)
    dwdy = gtx.zeros(cell_k_domain, allocator=b_end)
    diff_multfac_w = wpfloat("5.0")
    diff_multfac_n2w = gtx.as_field(k_domain, diff_multfac_n2w_arr, allocator=b_end)

    # Indices and parameters
    nrdmax = int32(13)
    interior_idx = int32(1)
    halo_idx = int32(min(5, grid.num_cells))
    horizontal_start = int32(0)
    horizontal_end = int32(grid.num_cells)
    vertical_start = int32(0)
    vertical_end = int32(levels)

    print("start")
    for _ in range(1000):
        apply_diffusion_to_w.with_backend(b_end)(
            area,
            geofac_n2s,
            w_old,
            w,
            diff_multfac_w,
            interior_idx,
            halo_idx,
            int32(grid.num_cells),
            horizontal_start,
            horizontal_end,
            vertical_start,
            vertical_end,
            offset_provider=grid.offset_providers,
        )
    print("end")
