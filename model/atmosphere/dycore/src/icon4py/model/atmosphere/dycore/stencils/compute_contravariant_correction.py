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
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_contravariant_correction(
    vn: fa.EdgeKField[wpfloat],
    ddxn_z_full: fa.EdgeKField[wpfloat],
    ddxt_z_full: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_35 or mo_velocity_advection_stencil_04."""
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


@program(grid_type=GridType.UNSTRUCTURED)
def compute_contravariant_correction(
    vn: fa.EdgeKField[wpfloat],
    ddxn_z_full: fa.EdgeKField[wpfloat],
    ddxt_z_full: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[wpfloat],
    z_w_concorr_me: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_contravariant_correction(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        out=z_w_concorr_me,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
