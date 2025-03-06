from icon4py.model.common.grid.e2c2v_sum import (
    E2C2V_simple_sum,
    E2C2V_smart_sum,
    E2C2V_smarter_sum,
)
from icon4py.model.common.grid.simple import SimpleGrid

import gt4py.next as gtx
from gt4py.next import Dimension
import numpy as np


def test_E2C2V_sums():
    grid = SimpleGrid()
    tol = 10e-6

    n_edges = grid.num_edges
    print("number of edges:", n_edges)
    n_vertices = grid.num_vertices
    print("number of vertices:", n_vertices)
    v_array = np.random.rand(n_vertices)
    print("vertex array:", v_array)

    vertex_domain = gtx.domain({Dimension("Vertex"): n_vertices})
    edge_domain = gtx.domain({Dimension("Edge"): n_edges})

    vertex_field = gtx.as_field(vertex_domain, v_array)
    print("vertex field:", vertex_field.asnumpy())

    edge_field_simple_sum = gtx.zeros(edge_domain)
    edge_field_smart_sum = gtx.zeros(edge_domain)
    edge_field_smarter_sum = gtx.zeros(edge_domain)

    E2C2V_simple_sum(
        vertex_field,
        edge_field_simple_sum,
        offset_provider=grid.offset_providers,
    )
    E2C2V_smart_sum(
        vertex_field,
        edge_field_smart_sum,
        offset_provider=grid.offset_providers,
    )
    E2C2V_smarter_sum(
        vertex_field,
        edge_field_smarter_sum,
        offset_provider=grid.offset_providers,
    )
    print(edge_field_simple_sum.asnumpy())
    print(edge_field_smart_sum.asnumpy())
    print(edge_field_smarter_sum.asnumpy())

    assert np.allclose(
        edge_field_simple_sum.asnumpy(), edge_field_smart_sum.asnumpy(), atol=tol
    )
    assert np.allclose(
        edge_field_simple_sum.asnumpy(), edge_field_smarter_sum.asnumpy(), atol=tol
    )
    assert np.allclose(
        edge_field_smart_sum.asnumpy(), edge_field_smarter_sum.asnumpy(), atol=tol
    )


test_E2C2V_sums()
print("Arrays have the same values")
