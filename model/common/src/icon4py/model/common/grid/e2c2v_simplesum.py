from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2V, E2C


@field_operator
def _E2C2V_simple_sum(
    vertex_input: fa.VertexKField[float],
) -> fa.EdgeKField[float]:
    return (
        vertex_input(C2V[0])(E2C[0])
        + vertex_input(C2V[1])(E2C[0])
        + vertex_input(C2V[1])(E2C[0])  # == vertex_input(C2V[0])(E2C[1])
        + vertex_input(C2V[2])(E2C[0])
        + vertex_input(C2V[2])(E2C[0])  # == vertex_input(C2V[1])(E2C[1])
        + vertex_input(C2V[2])(E2C[1])
    )


@program
def E2C2V_simple_sum(
    vertex_input: fa.VertexKField[float],
    edge_out: fa.EdgeKField[float],
):
    _E2C2V_simple_sum(vertex_input, out=edge_out)
