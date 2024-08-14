# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestApply2ndOrderDivergenceDamping(StencilTest):
    PROGRAM = apply_2nd_order_divergence_damping
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(grid, z_graddiv_vn: np.array, vn: np.array, scal_divdamp_o2, **kwargs) -> dict:
        vn = vn + (scal_divdamp_o2 * z_graddiv_vn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        z_graddiv_vn = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        scal_divdamp_o2 = wpfloat("5.0")

        return dict(
            z_graddiv_vn=z_graddiv_vn,
            vn=vn,
            scal_divdamp_o2=scal_divdamp_o2,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
