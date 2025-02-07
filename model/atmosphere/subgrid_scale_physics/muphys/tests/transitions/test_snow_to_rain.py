# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import snow_to_rain
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestSnowToRainDefault(StencilTest):
    PROGRAM = snow_to_rain
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, p: np.array, rho: np.array, dvsw0: np.array, qs: np.array, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 3.7268547760462804e-07))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = data_alloc.constant_field(grid, 275.83, dims.CellDim, dims.KDim, dtype=wpfloat),
            p                = data_alloc.constant_field(grid, 80134.5, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho              = data_alloc.constant_field(grid, 1.04892, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw0            = data_alloc.constant_field(grid, 0.00258631, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs               = data_alloc.constant_field(grid, 1.47687e-6, dims.CellDim, dims.KDim, dtype=wpfloat),
            conversion_rate  = data_alloc.constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )

