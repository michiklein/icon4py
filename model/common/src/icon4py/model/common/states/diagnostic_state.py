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

from dataclasses import dataclass

from gt4py.next.common import Field

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim


@dataclass
class DiagnosticState:
    """Class that contains the diagnostic state which is not used in dycore but may be used in physics.
    These variables are only stored for output purpose.

    Corresponds to ICON t_nh_diag
    """

    pressure: Field[[CellDim, KDim], float]
    temperature: Field[[CellDim, KDim], float]
    pressure_sfc: Field[[CellDim], float]
