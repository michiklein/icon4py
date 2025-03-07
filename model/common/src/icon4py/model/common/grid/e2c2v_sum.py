from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2V, E2C, E2V, E2C2V


@field_operator
def _E2C2V_simple_sum(
    vertex_input: fa.VertexField[float],
) -> fa.EdgeField[float]:
    return (
        vertex_input(C2V[0])(E2C[0])
        + vertex_input(C2V[0])(E2C[1])
        + vertex_input(C2V[1])(E2C[0])
        + vertex_input(C2V[1])(E2C[1])
        + vertex_input(C2V[2])(E2C[0])
        + vertex_input(C2V[2])(E2C[1])
    )


@program
def E2C2V_simple_sum(
    vertex_input: fa.VertexField[float],
    edge_out: fa.EdgeField[float],
):
    _E2C2V_simple_sum(vertex_input, out=edge_out)


@field_operator
def _E2C2V_smart_sum(
    vertex_input: fa.VertexField[float],
) -> fa.EdgeField[float]:
    return (
        vertex_input(C2V[0])(E2C[0])
        + vertex_input(C2V[0])(E2C[1])
        + vertex_input(C2V[1])(E2C[0])
        + vertex_input(C2V[1])(E2C[1])
        + vertex_input(C2V[2])(E2C[0])
        + vertex_input(C2V[2])(E2C[1])
    )


@program
def E2C2V_smart_sum(
    vertex_input: fa.VertexField[float],
    edge_out: fa.EdgeField[float],
):
    _E2C2V_smart_sum(vertex_input, out=edge_out)


@field_operator
def _E2C2V_smarter_sum(
    vertex_input: fa.VertexField[float],
) -> fa.EdgeField[float]:
    return (
        vertex_input(C2V[0])(E2C[0])
        + vertex_input(C2V[0])(E2C[1])
        + vertex_input(C2V[1])(E2C[0])
        + vertex_input(C2V[1])(E2C[1])
        + vertex_input(C2V[2])(E2C[0])
        + vertex_input(C2V[2])(E2C[1])
    )


@program
def E2C2V_smarter_sum(
    vertex_input: fa.VertexField[float],
    edge_out: fa.EdgeField[float],
):
    _E2C2V_smarter_sum(vertex_input, out=edge_out)


@field_operator
def _E2C2V_direct_sum(
    vertex_input: fa.VertexField[float],
) -> fa.EdgeField[float]:
    return (
        vertex_input(E2C2V[0])
        + vertex_input(E2C2V[1])
        + vertex_input(E2C2V[1])
        + vertex_input(E2C2V[2])
        + vertex_input(E2C2V[2])
        + vertex_input(E2C2V[3])
    )


@program
def E2C2V_direct_sum(
    vertex_input: fa.VertexField[float],
    edge_out: fa.EdgeField[float],
):
    _E2C2V_direct_sum(vertex_input, out=edge_out)
