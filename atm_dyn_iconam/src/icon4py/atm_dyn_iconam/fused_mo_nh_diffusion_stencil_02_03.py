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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field

from icon4py.atm_dyn_iconam.calculate_diagnostics_for_turbulance import (
    _calculate_diagnostics_for_turbulance,
)
from icon4py.atm_dyn_iconam.temporary_fields_for_turbulance_diagnostics import (
    _temporary_fields_for_turbulance_diagnostics,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_mo_nh_diffusion_stencil_02_03(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    diff_multfac_smag: Field[[KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    kh_c, div = _temporary_fields_for_turbulance_diagnostics(
        kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )
    div_ic, hdef_ic = _calculate_diagnostics_for_turbulance(div, kh_c, wgtfac_c)
    return div_ic, hdef_ic


@program
def fused_mo_nh_diffusion_stencil_02_03(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    diff_multfac_smag: Field[[KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _fused_mo_nh_diffusion_stencil_02_03(
        kh_smag_ec,
        vn,
        e_bln_c_s,
        geofac_div,
        diff_multfac_smag,
        wgtfac_c,
        out=(div_ic, hdef_ic),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
