# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.enhance_diffusion_coefficient_for_grid_point_cold_pools import (
    _enhance_diffusion_coefficient_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.temporary_field_for_grid_point_cold_pools_enhancement import (
    _temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    kh_smag_e: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    enh_diffu_3d = _temporary_field_for_grid_point_cold_pools_enhancement(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        smallest_vpfloat,
    )
    kh_smag_e_vp = _enhance_diffusion_coefficient_for_grid_point_cold_pools(kh_smag_e, enh_diffu_3d)
    return kh_smag_e_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    kh_smag_e: fa.EdgeKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        smallest_vpfloat,
        kh_smag_e,
        out=kh_smag_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
