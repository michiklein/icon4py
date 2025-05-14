import numpy as np
import cupy as cp
import sys
import os
sys.path.append('..')
from mklein_test import *
import gt4py.next as gtx
from gt4py.next import Dimension
from stencils_combined import c2e2c_sum_program
b_end = gtx.gtfn_gpu
xp = cp if "gpu" in str(b_end).lower() else np
grid_file = "../all_torus_files/big_torus_100000_100000_64.nc"
grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())
vertices, edges, cells = trim_grid(grid_file, grid)
vertices, edges, cells = reorder_trimmed_edges_and_cells(vertices, edges, cells, grid_file)
reorder_c2x(grid, grid_file, cells)
reorder_e2x(grid, grid_file, edges)
reorder_v2x(grid, grid_file, vertices)
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
cell_values = xp.asarray(rng.random(grid.num_cells))
cell_domain = gtx.domain({Dimension("Cell"): grid.num_cells})
cell_input = gtx.as_field(cell_domain, cell_values, allocator=b_end)
cell_output = gtx.zeros(cell_domain, allocator=b_end)
for _ in range(10000):
    c2e2c_sum_program(
        cell_input=cell_input,
        cell_out=cell_output,
        offset_provider=grid.offset_providers,
        num_edges=np.int64(len(edges)),
        num_cells=np.int64(len(cells))
    )