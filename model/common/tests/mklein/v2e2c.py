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
from stencils_combined import v2e2c_sum_program
from gt4py.next import int32
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
cell_values = xp.asarray(rng.random((grid.num_cells, levels)))

from icon4py.model.common.dimension import VertexDim, CellDim, KDim

vertex_domain = gtx.domain({VertexDim: grid.num_vertices, KDim: levels})
cell_domain = gtx.domain({CellDim: grid.num_cells, KDim: levels})

cell_input = gtx.as_field(cell_domain, cell_values, allocator=b_end)
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
    v2e2c_sum_program(
        cell_input=cell_input,
        vertex_out=vertex_output,
        offset_provider=grid.offset_providers,
        num_edges=int32(len(edges)),
        num_cells=int32(len(cells))
    )
print("end")