# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.apply_4th_order_divergence_damping import (
    apply_4th_order_divergence_damping,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestApply4thOrderDivergenceDamping(StencilTest):
    PROGRAM = apply_4th_order_divergence_damping
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        grid,
        scal_divdamp: np.array,
        z_graddiv2_vn: np.array,
        vn: np.array,
        **kwargs,
    ) -> dict:
        scal_divdamp = np.expand_dims(scal_divdamp, axis=0)
        vn = vn + (scal_divdamp * z_graddiv2_vn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        scal_divdamp = random_field(grid, dims.KDim, dtype=wpfloat)
        z_graddiv2_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            scal_divdamp=scal_divdamp,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
