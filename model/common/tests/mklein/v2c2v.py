# v2c2v.py
import numpy as np
import cupy as cp
from mklein_test import (
    get_torus_grid,
    trim_grid,
    reorder_trimmed_edges_and_cells,
    reorder_c2x,
    reorder_e2x,
    reorder_v2x,
    reindex_cells,
    reindex_edges,
    reindex_vertices,
    overwrite_chained_connectivities,
    ToZeroBasedIndexTransformation,
)
import gt4py.next as gtx
from gt4py.next import Dimension
from stencils_combined import v2c2v_sum_program

b_end = gtx.gtfn_gpu
xp = cp if "gpu" in str(b_end).lower() else np
grid_file = "../all_torus_files/torus_100000_100000_256.nc"
grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())
vertices, edges, cells = trim_grid(grid_file, grid)
vertices, edges, cells = reorder_trimmed_edges_and_cells(vertices, edges, cells, grid_file, grid)
reorder_c2x(grid, grid_file)
reorder_e2x(grid, grid_file)
reorder_v2x(grid, grid_file)
reindex_cells(grid, cells)
reindex_edges(grid, edges)
reindex_vertices(grid, vertices)
grid = overwrite_chained_connectivities(grid)

if xp.__name__ == "cupy":
    cp.get_default_memory_pool().free_all_blocks()
    for dim_name, connectivity in grid.connectivities.items():
        if isinstance(connectivity, np.ndarray):
            grid.connectivities[dim_name] = cp.asarray(connectivity)

rng = np.random.default_rng(1)
values = xp.asarray(rng.random(grid.num_vertices))
input_domain = gtx.domain({Dimension("Vertex"): grid.num_vertices})
output_domain = gtx.domain({Dimension("Vertex"): grid.num_vertices})
vertex_input = gtx.as_field(input_domain, values, allocator=b_end)
vertex_output = gtx.zeros(output_domain, allocator=b_end)

print("start")
for _ in range(10000):
    v2c2v_sum_program(
        vertex_input=vertex_input,
        vertex_output=vertex_output,
        offset_provider=grid.offset_providers,
        num_edges=np.int64(len(edges)),
        num_cells=np.int64(len(cells))
    )
print("end")
