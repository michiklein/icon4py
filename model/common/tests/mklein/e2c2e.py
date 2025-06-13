import numpy as np
import cupy as cp
from mklein_test import (
    get_torus_grid,
    trim_grid,
    reindex_cells,
    reindex_edges,
    reindex_vertices,
    ToZeroBasedIndexTransformation,
)
import gt4py.next as gtx
from stencils_combined import e2c2e_sum_program

b_end = gtx.gtfn_gpu
xp = cp if "gpu" in str(b_end).lower() else np

grid_file = "../all_torus_files/torus_100000_100000_256_reordered.nc"
levels = 80
grid = get_torus_grid(grid_file, levels, ToZeroBasedIndexTransformation())

vertices, edges, cells = trim_grid(grid_file, grid)

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
edge_values = xp.asarray(rng.random((grid.num_edges, levels)))

from icon4py.model.common.dimension import EdgeDim, KDim

edge_domain = gtx.domain({EdgeDim: grid.num_edges, KDim: levels})

edge_input = gtx.as_field(edge_domain, edge_values, allocator=b_end)
edge_output = gtx.zeros(edge_domain, allocator=b_end)

print("start")
for _ in range(1):
    e2c2e_sum_program(
        edge_input=edge_input,
        edge_output=edge_output,
        offset_provider=grid.offset_providers,
        num_edges=np.int64(len(edges)),
        num_cells=np.int64(len(cells))
    )
print("end")