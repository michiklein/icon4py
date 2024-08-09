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
from gt4py.next.ffront.fbuiltins import broadcast, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _init_two_edge_kdim_fields_with_zero_wp() -> (
    tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]
):
    """Formerly know as _mo_solve_nonhydro_stencil_14, _mo_solve_nonhydro_stencil_15, or _mo_solve_nonhydro_stencil_33."""
    return broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)), broadcast(
        wpfloat("0.0"), (dims.EdgeDim, dims.KDim)
    )


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_two_edge_kdim_fields_with_zero_wp(
    edge_kdim_field_with_zero_wp_1: fa.EdgeKField[wpfloat],
    edge_kdim_field_with_zero_wp_2: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _init_two_edge_kdim_fields_with_zero_wp(
        out=(edge_kdim_field_with_zero_wp_1, edge_kdim_field_with_zero_wp_2),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
