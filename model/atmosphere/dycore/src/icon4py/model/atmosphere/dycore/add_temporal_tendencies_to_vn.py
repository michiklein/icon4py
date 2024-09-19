# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_temporal_tendencies_to_vn(
    vn_nnow: fa.EdgeKField[wpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_24."""
    z_gradh_exner_wp = astype(z_gradh_exner, wpfloat)

    vn_nnew_wp = vn_nnow + dtime * (
        astype(ddt_vn_apc_ntl1, wpfloat)
        - cpd * z_theta_v_e * z_gradh_exner_wp
        + astype(ddt_vn_phy, wpfloat)
    )
    return vn_nnew_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_temporal_tendencies_to_vn(
    vn_nnow: fa.EdgeKField[wpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    vn_nnew: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _add_temporal_tendencies_to_vn(
        vn_nnow,
        ddt_vn_apc_ntl1,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        cpd,
        out=vn_nnew,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
