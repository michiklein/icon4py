import gt4py.next as gtx
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.math.stencils.operators import _compute_nabla2_on_cell


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_nabla2_on_cell(
    psi_c: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    nabla2_psi_c: fa.CellField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _compute_nabla2_on_cell(
        psi_c,
        geofac_n2s,
        out=nabla2_psi_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )