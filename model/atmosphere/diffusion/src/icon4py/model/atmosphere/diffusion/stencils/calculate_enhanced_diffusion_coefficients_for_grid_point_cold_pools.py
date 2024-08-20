# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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


# TODO: this will have to be removed once domain allows for imports
EdgeDim = dims.EdgeDim
KDim = dims.KDim


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
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
