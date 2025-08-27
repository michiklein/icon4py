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

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    _calculate_nabla2_for_z,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    _calculate_nabla2_of_theta,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim, E2C, E2CDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum


@field_operator
def _calculate_nabla2_for_theta(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
) -> fa.CellKField[vpfloat]:
    return astype(neighbor_sum((kh_smag_e * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0])))(C2E) * geofac_div(C2CE), axis=C2EDim), vpfloat)
    # return astype(neighbor_sum((kh_smag_e * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0])))(C2E) * geofac_div(C2CE), axis=C2EDim), vpfloat)
    # return astype(
    #     (
    #         (astype(kh_smag_e, wpfloat) * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0])))(C2E[0])
    #         * geofac_div(C2CE[0])
    #         + (astype(kh_smag_e, wpfloat) * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0])))(C2E[1])
    #         * geofac_div(C2CE[1])
    #         + (astype(kh_smag_e, wpfloat) * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0])))(C2E[2])
    #         * geofac_div(C2CE[2])
    #     ),
    #     vpfloat,
    # )

@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_for_theta(
    kh_smag_e: fa.EdgeKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_temp: fa.CellKField[vpfloat],
    num_cells: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_nabla2_for_theta(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v,
        geofac_div,
        out=z_temp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
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
    from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
    from gt4py.next import int32

    # Select backend (CPU/GPU)
    b_end = gtx.gtfn_gpu
    xp = cp if "gpu" in str(b_end).lower() else np

    parser = argparse.ArgumentParser(
        description="Run calculate_nabla2_for_theta with optional block sorting",
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
            if hasattr(provider, 'ndarray'):
                # Use skip_value = 0 for the C2E connectivity, keep -1 for the rest
                skip_val = None
                grid.offset_providers[name] = gtx.as_connectivity(
                    provider.domain,
                    codomain=provider.codomain,
                    data=provider.ndarray,
                    skip_value=skip_val,
                    allocator=b_end,
                )

    rng = np.random.default_rng(1)

    # ----- Allocate and wrap input / output fields -----
    kh_smag_e_arr = xp.asarray(rng.random((grid.num_edges, levels)))
    inv_dual_edge_length_arr = xp.asarray(rng.random((grid.num_edges,)))
    theta_v_arr = xp.asarray(rng.random((grid.num_cells, levels)))

    # The CE dimension length equals the number of edges per cell (triangular ⇒ 3)
    ce_len = grid.get_offset_provider("C2E").ndarray.shape[1]
    geofac_div_arr = xp.asarray(rng.random((grid.size[dims.CEDim],)))

    # Create domains for field allocation
    cell_k_domain = gtx.domain({CellDim: grid.num_cells, KDim: levels})
    edge_k_domain = gtx.domain({EdgeDim: grid.num_edges, KDim: levels})
    edge_domain = gtx.domain({EdgeDim: grid.num_edges})
    geofac_div_domain = gtx.domain({dims.CEDim: grid.size[dims.CEDim]})

    kh_smag_e = gtx.as_field(edge_k_domain, kh_smag_e_arr, allocator=b_end)
    inv_dual_edge_length = gtx.as_field(edge_domain, inv_dual_edge_length_arr, allocator=b_end)
    theta_v = gtx.as_field(cell_k_domain, theta_v_arr, allocator=b_end)
    geofac_div = gtx.as_field(geofac_div_domain, geofac_div_arr, allocator=b_end)
    z_temp = gtx.zeros(cell_k_domain, allocator=b_end)

    horizontal_start = int32(0)
    horizontal_end = int32(grid.num_cells)
    vertical_start = int32(0)
    vertical_end = int32(levels)

    print("start")
    for _ in range(1000):
        calculate_nabla2_for_theta.with_backend(b_end)(
            kh_smag_e,
            inv_dual_edge_length,
            theta_v,
            geofac_div,
            z_temp,
            int32(grid.num_cells),
            horizontal_start,
            horizontal_end,
            vertical_start,
            vertical_end,
            offset_provider=grid.offset_providers,
        )
    print("end")
