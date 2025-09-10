#!/usr/bin/env python3

import numpy as np
import cupy as cp
from icon4py.model.common.grid.grid_manager import GridManager, ToZeroBasedIndexTransformation
from icon4py.model.common.grid.vertical import VerticalGridConfig
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim
import gt4py.next as gtx

b_end = gtx.gtfn_gpu
xp = cp

@field_operator
def test_edge_conditional(edge_input: fa.EdgeKField[float], num_edges: int) -> fa.EdgeKField[float]:
    return concat_where(
        EdgeDim < num_edges // np.int64(3),
        edge_input * 2.0,  # East pattern (first third)
        concat_where(
            EdgeDim < np.int64(2) * (num_edges // np.int64(3)),
            edge_input * 3.0,  # North pattern (middle third) 
            edge_input * 4.0   # Southeast pattern (last third)
        )
    )

@program(backend=b_end)
def test_program(edge_input: fa.EdgeKField[float], edge_out: fa.EdgeKField[float], num_edges: int):
    test_edge_conditional(edge_input, num_edges, out=edge_out)

if __name__ == "__main__":
    grid_file = "../all_torus_files/torus_100000_100000_256_reordered.nc"
    grid_manager = GridManager(ToZeroBasedIndexTransformation(), grid_file, VerticalGridConfig(1))
    grid_manager(None)
    grid = grid_manager.grid
    
    # Convert connectivities to GPU
    for name, provider in grid.offset_providers.items():
        if hasattr(provider, 'ndarray'):
            grid.offset_providers[name] = gtx.as_connectivity(
                provider.domain, codomain=provider.codomain, 
                data=provider.ndarray, skip_value=-1, allocator=b_end
            )
    
    # Use all edges from the actual grid
    num_edges = grid.num_edges
    domain = gtx.domain({EdgeDim: num_edges, KDim: 1})
    edge_data = xp.ones((num_edges, 1))
    
    input_field = gtx.as_field(domain, edge_data, allocator=b_end)
    output_field = gtx.zeros(domain, allocator=b_end)
    
    test_program(
        edge_input=input_field,
        edge_out=output_field,
        num_edges=np.int64(num_edges),  # Pass actual number of edges at runtime
        offset_provider=grid.offset_providers
    )
    
    result = output_field.ndarray
    print(f"Total edges: {num_edges}")
    print(f"Threshold 1 (num_edges//3): {num_edges//3}")
    print(f"Threshold 2 (2*num_edges//3): {2*(num_edges//3)}")
    print(f"All results: {result[:, 0]}")
    print(f"Expected: first {num_edges//3} should be 2.0, next {num_edges//3} should be 3.0, rest should be 4.0")