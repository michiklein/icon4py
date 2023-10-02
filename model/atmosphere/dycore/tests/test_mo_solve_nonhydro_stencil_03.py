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

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_03 import (
    mo_solve_nonhydro_stencil_03,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil03(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_03
    OUTPUTS = ("z_exner_ex_pr",)

    @staticmethod
    def reference(mesh, z_exner_ex_pr: np.array, **kwargs) -> dict:
        z_exner_ex_pr = np.zeros_like(z_exner_ex_pr)
        return dict(z_exner_ex_pr=z_exner_ex_pr)

    @pytest.fixture
    def input_data(self, mesh):
        z_exner_ex_pr = random_field(mesh, CellDim, KDim)

        return dict(
            z_exner_ex_pr=z_exner_ex_pr,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_cells),
            vertical_start=int32(1),
            vertical_end=int32(mesh.k_level),
        )
