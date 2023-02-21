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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import C2E, C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _mo_velocity_advection_stencil_08(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
) -> Field[[CellDim, KDim], float]:
    z_ekinh = neighbor_sum(z_kin_hor_e(C2E) * e_bln_c_s, axis=C2EDim)
    return z_ekinh


@program
def mo_velocity_advection_stencil_08(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
):
    _mo_velocity_advection_stencil_08(z_kin_hor_e, e_bln_c_s, out=z_ekinh)
