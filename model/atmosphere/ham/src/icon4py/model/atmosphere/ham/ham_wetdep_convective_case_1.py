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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_wp import (
    _set_cell_kdim_field_to_zero_wp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _ham_wetdep_convective_case_1(
    pmfu: Field[[CellDim, KDim], wpfloat],
) -> (
    tuple[Field[[CellDim, KDim], wpfloat],
          Field[[CellDim, KDim], wpfloat]]
):

    return (pmfu, _set_cell_kdim_field_to_zero_wp())


@program(grid_type=GridType.UNSTRUCTURED)
def ham_wetdep_convective_case_1(
    pmfu            : Field[[CellDim, KDim], wpfloat],
    zmf             : Field[[CellDim, KDim], wpfloat],
    zxtp10          : Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end  : int32,
    vertical_start  : int32,
    vertical_end    : int32
):

    _ham_wetdep_convective_case_1( pmfu,
                                   out = (zmf, zxtp10),
                                   domain = {
                                       CellDim: (horizontal_start, horizontal_end),
                                       KDim: (vertical_start, vertical_end)
                                   }
    )