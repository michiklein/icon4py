from icon4py.model.common.grid.e2c2v_sum import (
    E2C2V_simple_sum,
    E2C2V_smart_sum,
    E2C2V_smarter_sum,
    E2C2V_direct_sum,
)
from icon4py.model.common.grid.simple import SimpleGrid

import gt4py.next as gtx
from gt4py.next import Dimension
from icon4py.model.common import model_backends

import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig


def init_grid_manager(
    fname, num_levels=65, transformation=ToZeroBasedIndexTransformation()
):
    grid_manager = GridManager(
        transformation,
        fname,
        VerticalGridConfig(num_levels),
    )
    grid_manager(None)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    simple_grid = grid_manager.grid
    return simple_grid


def test_E2C2V_sums():
    grid_file = input("Path to grid file: ")
    grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())
    tol = 10e-6

    n_edges = grid.num_edges
    n_vertices = grid.num_vertices
    v_array = np.random.rand(n_vertices)
    print("vertex array:", v_array)

    vertex_domain = gtx.domain({Dimension("Vertex"): n_vertices})
    edge_domain = gtx.domain({Dimension("Edge"): n_edges})

    vertex_field = gtx.as_field(vertex_domain, v_array)
    print("vertex field:", vertex_field.asnumpy())

    edge_field_simple_sum = gtx.zeros(edge_domain)
    edge_field_smart_sum = gtx.zeros(edge_domain)
    edge_field_smarter_sum = gtx.zeros(edge_domain)
    edge_field_direct_sum = gtx.zeros(edge_domain)

    # E2C2V_simple_sum.with_backend(model_backends.BACKENDS["gtfn_cpu"])(
    #     vertex_field,
    #     edge_field_simple_sum,
    #     offset_provider=grid.offset_providers,
    # )
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
    E2C2V_direct_sum(
        vertex_field,
        edge_field_direct_sum,
        offset_provider=grid.offset_providers,
    )
    print("direct:", edge_field_direct_sum.asnumpy())
    print("simple:", edge_field_simple_sum.asnumpy())
    print("smart:", edge_field_smart_sum.asnumpy())
    print("smarter:", edge_field_smarter_sum.asnumpy())

    assert np.allclose(
        edge_field_simple_sum.asnumpy(), edge_field_direct_sum.asnumpy(), atol=tol
    )
    assert np.allclose(
        edge_field_simple_sum.asnumpy(), edge_field_smart_sum.asnumpy(), atol=tol
    )
    assert np.allclose(
        edge_field_simple_sum.asnumpy(), edge_field_smarter_sum.asnumpy(), atol=tol
    )
    assert np.allclose(
        edge_field_smart_sum.asnumpy(), edge_field_smarter_sum.asnumpy(), atol=tol
    )

    assert np.allclose(
        edge_field_smart_sum.asnumpy(), edge_field_direct_sum.asnumpy(), atol=tol
    )
    assert np.allclose(
        edge_field_smarter_sum.asnumpy(), edge_field_direct_sum.asnumpy(), atol=tol
    )


test_E2C2V_sums()
print("Arrays have the same values")
