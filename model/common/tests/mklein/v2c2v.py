import numpy as np
import cupy as cp
import argparse
import time
from mklein_test import (
    get_torus_grid,
    trim_grid,
    reindex_cells,
    reindex_edges,
    reindex_vertices,
    sort_into_blocks_32,
    ToZeroBasedIndexTransformation,
)
import gt4py.next as gtx
from stencils_combined import v2c2v_sum_program
from gt4py.next import int32
b_end = gtx.gtfn_gpu
xp = cp if "gpu" in str(b_end).lower() else np

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run v2c2v with optional block sorting')
    parser.add_argument('--block-sort', action='store_true', 
                       help='Sort edges and cells into blocks of 32')
    args = parser.parse_args()

    grid_file = "../all_torus_files/torus_100000_100000_256_reordered.nc"
    levels = 80
    load_start = time.perf_counter()
    grid = get_torus_grid(grid_file, levels, ToZeroBasedIndexTransformation())
    print(f"Grid loaded in {time.perf_counter() - load_start:.6f}s")

    vertices, edges, cells = trim_grid(grid_file, grid)

    if args.block_sort:
        # Use block sorting instead of regular reindexing
        sort_into_blocks_32(grid, grid_file, edges, cells)
        reindex_vertices(grid, vertices)
    else:
        # Use original reindexing
        reindex_cells(grid, cells)
        reindex_edges(grid, edges)
        reindex_vertices(grid, vertices)

    if xp.__name__ == "cupy":
        cp.get_default_memory_pool().free_all_blocks()
        for name, connectivity in grid.connectivities.items():
            if hasattr(connectivity, 'ndarray'):
                connectivity.ndarray = cp.asarray(connectivity.ndarray)
            else:
                grid.connectivities[name] = cp.asarray(connectivity)

    rng = np.random.default_rng(1)
    vertex_values = xp.asarray(rng.random((grid.num_vertices, levels)))

    from icon4py.model.common.dimension import VertexDim, KDim

    vertex_domain = gtx.domain({VertexDim: grid.num_vertices, KDim: levels})

    vertex_input = gtx.as_field(vertex_domain, vertex_values, allocator=b_end)
    vertex_output = gtx.zeros(vertex_domain, allocator=b_end)
    if xp.__name__ == "cupy":
            for name, provider in grid.offset_providers.items():
                if hasattr(provider, 'ndarray'):
                    grid.offset_providers[name] = gtx.as_connectivity(
                        provider.domain,
                        codomain=provider.codomain, 
                        data=provider.ndarray, 
                        skip_value=-1,
                        allocator=b_end
                    )
                    
    print("start")
    for _ in range(1):
        v2c2v_sum_program(
            vertex_input=vertex_input,
            vertex_out=vertex_output,
            offset_provider=grid.offset_providers,
            num_edges=int32(len(edges)),
            num_cells=int32(len(cells))
        )
    print("end")