# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import rain_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest

class TestRainToGraupel1(StencilTest):
    PROGRAM = rain_to_graupel
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qc: np.array, qr: np.array, qi: np.array, qs: np.array, mi: np.array, dvsw: np.array, dt: wpfloat, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 5.1570340525841922e-17))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t       = data_alloc.constant_field(grid, 258.542, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho     = data_alloc.constant_field(grid, 0.956089, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc      = data_alloc.constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr      = data_alloc.constant_field(grid, 3.01332e-11, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi      = data_alloc.constant_field(grid, 5.57166e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs      = data_alloc.constant_field(grid, 3.55432e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            mi      = data_alloc.constant_field(grid, 1.0e-9, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw    = data_alloc.constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt      = 30.0,
            conversion_rate = data_alloc.constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )
