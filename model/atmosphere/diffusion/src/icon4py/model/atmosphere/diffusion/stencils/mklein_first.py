# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2V, C2VDim, E2C, E2CDim
from gt4py.next.ffront.fbuiltins import neighbor_sum


@field_operator
def _c2v_stencil(
    vertex_input: fa.VertexKField[float]
) -> fa.CellKField[float]:
    return neighbor_sum(vertex_input(C2V), axis=gtx.Dims[C2VDim])

@field_operator
def _e2c_stencil(
    cell_input: fa.CellKField[float]
) -> fa.EdgeKField[float]:
    return neighbor_sum(cell_input(E2C), axis=gtx.Dims[E2CDim])

@field_operator
def _e2c2v_stencil(
    vertex_input: fa.VertexKField[float]
) -> fa.EdgeKField[float]:
    tmp = _c2v_stencil(vertex_input)
    return _e2c_stencil(tmp)

@program(grid_type=GridType.UNSTRUCTURED)
def c2v_stecil(
    vertex_input: fa.VertexKField[float]
):
     _c2v_stencil(vertex_input)   

@program
def e2c_stencil(
    cell_input: fa.CellKField[float]
): 
    _e2c_stencil(cell_input)

@program
def e2c2v_stencil(
    vertex_input: fa.VertexKField[float]
):
    _e2c2v_stencil(vertex_input)
    