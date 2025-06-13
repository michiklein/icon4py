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
from stencils_combined import v2e2v_sum_program

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
vertex_values = xp.asarray(rng.random((grid.num_vertices, levels)))

from icon4py.model.common.dimension import VertexDim, KDim

vertex_domain = gtx.domain({VertexDim: grid.num_vertices, KDim: levels})

vertex_input = gtx.as_field(vertex_domain, vertex_values, allocator=b_end)
vertex_output = gtx.zeros(vertex_domain, allocator=b_end)

print("start")
for _ in range(1):
    v2e2v_sum_program(
        vertex_input=vertex_input,
        vertex_out=vertex_output,
        offset_provider=grid.offset_providers,
        num_edges=np.int64(len(edges)),
        num_cells=np.int64(len(cells))
    )
print("end")