from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2V, C2E, V2C, V2E, E2C, E2V
import gt4py.next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners import dace as dace_be

b_end = dace_be.run_dace_gpu


# ------- V STENCILS -------
@field_operator
def _v2c2e_sum(edge_input: fa.EdgeField[float]) -> fa.VertexField[float]:
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
    edge_input: fa.EdgeField[float],
    vertex_out: fa.VertexField[float],
    num_cells: int,
    num_edges: int,
):
    _v2c2e_sum(edge_input, out=vertex_out)


@field_operator
def _v2c2v_sum(vertex_input: fa.VertexField[float]) -> fa.VertexField[float]:
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
    vertex_input: fa.VertexField[float],
    vertex_out: fa.VertexField[float],
    num_cells: int,
    num_edges: int,
):
    _v2c2v_sum(vertex_input, out=vertex_out)


@field_operator
def _v2e2c_sum(cell_input: fa.CellField[float]) -> fa.VertexField[float]:
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
    cell_input: fa.CellField[float],
    vertex_out: fa.VertexField[float],
    num_cells: int,
    num_edges: int,
):
    _v2e2c_sum(cell_input, out=vertex_out)


@field_operator
def _v2e2v_sum(vertex_input: fa.VertexField[float]) -> fa.VertexField[float]:
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
    vertex_input: fa.VertexField[float],
    vertex_out: fa.VertexField[float],
    num_cells: int,
    num_edges: int,
):
    _v2e2v_sum(vertex_input, out=vertex_out)


# ------- E STENCILS -------
@field_operator
def _e2c2e_sum(edge_input: fa.EdgeField[float]) -> fa.EdgeField[float]:
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
    edge_input: fa.EdgeField[float],
    edge_out: fa.EdgeField[float],
    num_cells: int,
    num_edges: int,
):
    _e2c2e_sum(edge_input, out=edge_out)


@field_operator
def _e2c2v_sum(vertex_input: fa.VertexField[float]) -> fa.EdgeField[float]:
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
    vertex_input: fa.VertexField[float],
    edge_out: fa.EdgeField[float],
    num_cells: int,
    num_edges: int,
):
    _e2c2v_sum(vertex_input, out=edge_out)


@field_operator
def _e2v2c_sum(cell_input: fa.CellField[float]) -> fa.EdgeField[float]:
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
    cell_input: fa.CellField[float],
    edge_out: fa.EdgeField[float],
    num_cells: int,
    num_edges: int,
):
    _e2v2c_sum(cell_input, out=edge_out)


@field_operator
def _e2v2e_sum(edge_input: fa.EdgeField[float]) -> fa.EdgeField[float]:
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
    edge_input: fa.EdgeField[float],
    edge_out: fa.EdgeField[float],
    num_cells: int,
    num_edges: int,
):
    _e2v2e_sum(edge_input, out=edge_out)


# ------- C STENCILS -------
@field_operator
def _c2e2c_sum(cell_input: fa.CellField[float]) -> fa.CellField[float]:
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
    cell_input: fa.CellField[float],
    cell_out: fa.CellField[float],
    num_cells: int,
    num_edges: int,
):
    _c2e2c_sum(cell_input, out=cell_out)


@field_operator
def _c2e2v_sum(vertex_input: fa.VertexField[float]) -> fa.CellField[float]:
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
    vertex_input: fa.VertexField[float],
    cell_out: fa.CellField[float],
    num_cells: int,
    num_edges: int,
):
    _c2e2v_sum(vertex_input, out=cell_out)


@field_operator
def _c2v2c_sum(cell_input: fa.CellField[float]) -> fa.CellField[float]:
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
    cell_input: fa.CellField[float],
    cell_out: fa.CellField[float],
    num_cells: int,
    num_edges: int,
):
    _c2v2c_sum(cell_input, out=cell_out)


@field_operator
def _c2v2e_sum(edge_input: fa.EdgeField[float]) -> fa.CellField[float]:
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
    edge_input: fa.EdgeField[float],
    cell_out: fa.CellField[float],
    num_cells: int,
    num_edges: int,
):
    _c2v2e_sum(edge_input, out=cell_out)
