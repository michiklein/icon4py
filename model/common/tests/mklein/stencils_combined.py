from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2V, C2E, V2C, V2E, E2C, E2V
import gt4py.next as gtx
from gt4py.next.program_processors.runners.dace import (
        run_dace_gpu_cached as run_dace_gpu,
    )
from gt4py.next.ffront.experimental import concat_where
from icon4py.model.common import dimension as dims
from gt4py.next import int32

b_end = gtx.gtfn_gpu
# b_end = run_dace_gpu


# ------- V STENCILS -------
@field_operator
def _v2c2e_sum(edge_input: fa.EdgeKField[float]) -> fa.VertexKField[float]:
    return (
        edge_input(C2E[0])(V2C[0])
        + edge_input(C2E[1])(V2C[0])
        + edge_input(C2E[2])(V2C[0])
        + edge_input(C2E[0])(V2C[1])
        + edge_input(C2E[1])(V2C[1])
        + edge_input(C2E[2])(V2C[1])
        + edge_input(C2E[0])(V2C[2])
        + edge_input(C2E[1])(V2C[2])
        + edge_input(C2E[2])(V2C[2])
        + edge_input(C2E[0])(V2C[3])
        + edge_input(C2E[1])(V2C[3])
        + edge_input(C2E[2])(V2C[3])
        + edge_input(C2E[0])(V2C[4])
        + edge_input(C2E[1])(V2C[4])
        + edge_input(C2E[2])(V2C[4])
        + edge_input(C2E[0])(V2C[5])
        + edge_input(C2E[1])(V2C[5])
        + edge_input(C2E[2])(V2C[5])
    )


@program(backend=b_end)
def v2c2e_sum_program(
    edge_input: fa.EdgeKField[float],
    vertex_out: fa.VertexKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _v2c2e_sum(edge_input, out=vertex_out)


@field_operator
def _v2c2v_sum(vertex_input: fa.VertexKField[float]) -> fa.VertexKField[float]:
    return (
        vertex_input(C2V[0])(V2C[0])
        + vertex_input(C2V[1])(V2C[0])
        + vertex_input(C2V[2])(V2C[0])
        + vertex_input(C2V[0])(V2C[1])
        + vertex_input(C2V[1])(V2C[1])
        + vertex_input(C2V[2])(V2C[1])
        + vertex_input(C2V[0])(V2C[2])
        + vertex_input(C2V[1])(V2C[2])
        + vertex_input(C2V[2])(V2C[2])
        + vertex_input(C2V[0])(V2C[3])
        + vertex_input(C2V[1])(V2C[3])
        + vertex_input(C2V[2])(V2C[3])
        + vertex_input(C2V[0])(V2C[4])
        + vertex_input(C2V[1])(V2C[4])
        + vertex_input(C2V[2])(V2C[4])
        + vertex_input(C2V[0])(V2C[5])
        + vertex_input(C2V[1])(V2C[5])
        + vertex_input(C2V[2])(V2C[5])
    )


@program(backend=b_end)
def v2c2v_sum_program(
    vertex_input: fa.VertexKField[float],
    vertex_out: fa.VertexKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _v2c2v_sum(vertex_input, out=vertex_out)


@field_operator
def _v2e2c_sum(cell_input: fa.CellKField[float]) -> fa.VertexKField[float]:
    return (
        cell_input(E2C[0])(V2E[0])
        + cell_input(E2C[1])(V2E[0])
        + cell_input(E2C[0])(V2E[1])
        + cell_input(E2C[1])(V2E[1])
        + cell_input(E2C[0])(V2E[2])
        + cell_input(E2C[1])(V2E[2])
        + cell_input(E2C[0])(V2E[3])
        + cell_input(E2C[1])(V2E[3])
        + cell_input(E2C[0])(V2E[4])
        + cell_input(E2C[1])(V2E[4])
        + cell_input(E2C[0])(V2E[5])
        + cell_input(E2C[1])(V2E[5])
    )


@program(backend=b_end)
def v2e2c_sum_program(
    cell_input: fa.CellKField[float],
    vertex_out: fa.VertexKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _v2e2c_sum(cell_input, out=vertex_out)


@field_operator
def _v2e2v_sum(vertex_input: fa.VertexKField[float]) -> fa.VertexKField[float]:
    return (
        vertex_input(E2V[0])(V2E[0])
        + vertex_input(E2V[1])(V2E[0])
        + vertex_input(E2V[0])(V2E[1])
        + vertex_input(E2V[1])(V2E[1])
        + vertex_input(E2V[0])(V2E[2])
        + vertex_input(E2V[1])(V2E[2])
        + vertex_input(E2V[0])(V2E[3])
        + vertex_input(E2V[1])(V2E[3])
        + vertex_input(E2V[0])(V2E[4])
        + vertex_input(E2V[1])(V2E[4])
        + vertex_input(E2V[0])(V2E[5])
        + vertex_input(E2V[1])(V2E[5])
    )


@program(backend=b_end)
def v2e2v_sum_program(
    vertex_input: fa.VertexKField[float],
    vertex_out: fa.VertexKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _v2e2v_sum(vertex_input, out=vertex_out)


# ------- E STENCILS -------
@field_operator
def _e2c2e_sum(edge_input: fa.EdgeKField[float], num_edges: int32) -> fa.EdgeKField[float]:
    return (
        edge_input(C2E[0])(E2C[0])
        + edge_input(C2E[1])(E2C[0])
        + edge_input(C2E[2])(E2C[0])
        + edge_input(C2E[0])(E2C[1])
        + edge_input(C2E[1])(E2C[1])
        + edge_input(C2E[2])(E2C[1])
    )


@program(backend=b_end)
def e2c2e_sum_program(
    edge_input: fa.EdgeKField[float],
    edge_out: fa.EdgeKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _e2c2e_sum(edge_input, num_edges, out=edge_out)


@field_operator
def _e2c2v_sum(vertex_input: fa.VertexKField[float], num_edges: int32) -> fa.EdgeKField[float]:
    return (
        vertex_input(C2V[0])(E2C[0])
        + vertex_input(C2V[1])(E2C[0])
        + vertex_input(C2V[2])(E2C[0])
        + vertex_input(C2V[0])(E2C[1])
        + vertex_input(C2V[1])(E2C[1])
        + vertex_input(C2V[2])(E2C[1])
    )


@program(backend=b_end)
def e2c2v_sum_program(
    vertex_input: fa.VertexKField[float],
    edge_out: fa.EdgeKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _e2c2v_sum(vertex_input, num_edges, out=edge_out)


@field_operator
def _e2v2c_sum(cell_input: fa.CellKField[float], num_edges: int32) -> fa.EdgeKField[float]:
    return (
        cell_input(V2C[0])(E2V[0])
        + cell_input(V2C[1])(E2V[0])
        + cell_input(V2C[2])(E2V[0])
        + cell_input(V2C[3])(E2V[0])
        + cell_input(V2C[4])(E2V[0])
        + cell_input(V2C[5])(E2V[0])
        + cell_input(V2C[0])(E2V[1])
        + cell_input(V2C[1])(E2V[1])
        + cell_input(V2C[2])(E2V[1])
        + cell_input(V2C[3])(E2V[1])
        + cell_input(V2C[4])(E2V[1])
        + cell_input(V2C[5])(E2V[1])
    )


@program(backend=b_end)
def e2v2c_sum_program(
    cell_input: fa.CellKField[float],
    edge_out: fa.EdgeKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _e2v2c_sum(cell_input, num_cells, out=edge_out)


@field_operator
def _e2v2e_sum(edge_input: fa.EdgeKField[float], num_edges: int32) -> fa.EdgeKField[float]:
    return (
        edge_input(V2E[0])(E2V[0])
        + edge_input(V2E[1])(E2V[0])
        + edge_input(V2E[2])(E2V[0])
        + edge_input(V2E[3])(E2V[0])
        + edge_input(V2E[4])(E2V[0])
        + edge_input(V2E[5])(E2V[0])
        + edge_input(V2E[0])(E2V[1])
        + edge_input(V2E[1])(E2V[1])
        + edge_input(V2E[2])(E2V[1])
        + edge_input(V2E[3])(E2V[1])
        + edge_input(V2E[4])(E2V[1])
        + edge_input(V2E[5])(E2V[1])
    )


@program(backend=b_end)
def e2v2e_sum_program(
    edge_input: fa.EdgeKField[float],
    edge_out: fa.EdgeKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _e2v2e_sum(edge_input, num_edges, out=edge_out)


# ------- C STENCILS -------
@field_operator
def _c2e2c_sum(cell_input: fa.CellKField[float], num_cells: int32) -> fa.CellKField[float]:
    return (
        cell_input(E2C[0])(C2E[0])
        + cell_input(E2C[1])(C2E[0])
        + cell_input(E2C[0])(C2E[1])
        + cell_input(E2C[1])(C2E[1])
        + cell_input(E2C[0])(C2E[2])
        + cell_input(E2C[1])(C2E[2])
    )


@program(backend=b_end)
def c2e2c_sum_program(
    cell_input: fa.CellKField[float],
    cell_out: fa.CellKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _c2e2c_sum(cell_input, num_cells, out=cell_out)


@field_operator
def _c2e2v_sum(vertex_input: fa.VertexKField[float], num_cells: int32) -> fa.CellKField[float]:
    return (
        vertex_input(E2V[0])(C2E[0])
        + vertex_input(E2V[1])(C2E[0])
        + vertex_input(E2V[0])(C2E[1])
        + vertex_input(E2V[1])(C2E[1])
        + vertex_input(E2V[0])(C2E[2])
        + vertex_input(E2V[1])(C2E[2])
    )


@program(backend=b_end)
def c2e2v_sum_program(
    vertex_input: fa.VertexKField[float],
    cell_out: fa.CellKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _c2e2v_sum(vertex_input, num_cells, out=cell_out)


@field_operator
def _c2v2c_sum(cell_input: fa.CellKField[float], num_cells: int32) -> fa.CellKField[float]:
    return (
        cell_input(V2C[0])(C2V[0])
        + cell_input(V2C[1])(C2V[0])
        + cell_input(V2C[2])(C2V[0])
        + cell_input(V2C[3])(C2V[0])
        + cell_input(V2C[4])(C2V[0])
        + cell_input(V2C[5])(C2V[0])
        + cell_input(V2C[0])(C2V[1])
        + cell_input(V2C[1])(C2V[1])
        + cell_input(V2C[2])(C2V[1])
        + cell_input(V2C[3])(C2V[1])
        + cell_input(V2C[4])(C2V[1])
        + cell_input(V2C[5])(C2V[1])
        + cell_input(V2C[0])(C2V[2])
        + cell_input(V2C[1])(C2V[2])
        + cell_input(V2C[2])(C2V[2])
        + cell_input(V2C[3])(C2V[2])
        + cell_input(V2C[4])(C2V[2])
        + cell_input(V2C[5])(C2V[2])
    )


@program(backend=b_end)
def c2v2c_sum_program(
    cell_input: fa.CellKField[float],
    cell_out: fa.CellKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _c2v2c_sum(cell_input, num_cells, out=cell_out)


@field_operator
def _c2v2e_sum(edge_input: fa.EdgeKField[float], num_cells: int32) -> fa.CellKField[float]:
    return (
        edge_input(V2E[0])(C2V[0])
        + edge_input(V2E[1])(C2V[0])
        + edge_input(V2E[2])(C2V[0])
        + edge_input(V2E[3])(C2V[0])
        + edge_input(V2E[4])(C2V[0])
        + edge_input(V2E[5])(C2V[0])
        + edge_input(V2E[0])(C2V[1])
        + edge_input(V2E[1])(C2V[1])
        + edge_input(V2E[2])(C2V[1])
        + edge_input(V2E[3])(C2V[1])
        + edge_input(V2E[4])(C2V[1])
        + edge_input(V2E[5])(C2V[1])
        + edge_input(V2E[0])(C2V[2])
        + edge_input(V2E[1])(C2V[2])
        + edge_input(V2E[2])(C2V[2])
        + edge_input(V2E[3])(C2V[2])
        + edge_input(V2E[4])(C2V[2])
        + edge_input(V2E[5])(C2V[2])
    )


@program(backend=b_end)
def c2v2e_sum_program(
    edge_input: fa.EdgeKField[float],
    cell_out: fa.CellKField[float],
    num_cells: int32,
    num_edges: int32,
):
    _c2v2e_sum(edge_input, num_cells, out=cell_out)
