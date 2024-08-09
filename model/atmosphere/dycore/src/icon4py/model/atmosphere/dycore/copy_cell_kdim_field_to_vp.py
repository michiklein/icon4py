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
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _copy_cell_kdim_field_to_vp(
    field: fa.CellKField[wpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_11 or _mo_solve_nonhydro_stencil_59."""
    field_copy = astype(field, vpfloat)
    return field_copy


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def copy_cell_kdim_field_to_vp(
    field: fa.CellKField[wpfloat],
    field_copy: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _copy_cell_kdim_field_to_vp(
        field,
        out=field_copy,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
