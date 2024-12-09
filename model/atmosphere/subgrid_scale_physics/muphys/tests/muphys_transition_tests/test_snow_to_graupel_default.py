# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.snow_to_graupel import snow_to_graupel
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestSnowToGraupelDefault(StencilTest):
    PROGRAM = snow_to_graupel
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(grid, t: np.array, rho: np.array, qc: np.array, qs: np.array, QMIN: wpfloat, TFRZ_HOM: wpfloat, **kwargs) -> dict:
        return dict(conversion_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 281.157, dims.CellDim, dtype=wpfloat),
            rho              = constant_field(grid, 0.93171, dims.CellDim, dtype=wpfloat),
            qc               = constant_field(grid, 0.0, dims.CellDim, dtype=wpfloat),
            qs               = constant_field(grid, 4.34854e-5, dims.CellDim, dtype=wpfloat),
            QMIN             = graupel_ct.qmin,
            TFRZ_HOM         = graupel_ct.tfrz_hom,
            conversion_rate  = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )

