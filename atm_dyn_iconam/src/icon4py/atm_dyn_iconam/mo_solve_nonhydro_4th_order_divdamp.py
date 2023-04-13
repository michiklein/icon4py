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
from gt4py.next.ffront.fbuiltins import Field, broadcast
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_4th_order_divdamp(
    scal_divdamp: Field[[KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    scal_divdamp = broadcast(scal_divdamp, (EdgeDim, KDim))
    vn = vn + (scal_divdamp * z_graddiv2_vn)
    return vn


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_4th_order_divdamp(
    scal_divdamp: Field[[KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _mo_solve_nonhydro_4th_order_divdamp(
        scal_divdamp,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
