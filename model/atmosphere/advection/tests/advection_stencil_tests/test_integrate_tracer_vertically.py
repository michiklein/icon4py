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

import icon4py.model.testing.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_vertically import (
    integrate_tracer_vertically,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


class TestIntegrateTracerVertically(helpers.StencilTest):
    PROGRAM = integrate_tracer_vertically
    OUTPUTS = ("tracer_new",)

    @staticmethod
    def reference(
        grid,
        tracer_now: np.array,
        rhodz_now: np.array,
        p_mflx_tracer_v: np.array,
        deepatmo_divzl: np.array,
        deepatmo_divzu: np.array,
        rhodz_new: np.array,
        k: np.array,
        ivadv_tracer: gtx.int32,
        iadv_slev_jt: gtx.int32,
        p_dtime: float,
        **kwargs,
    ) -> dict:
        if ivadv_tracer != 0:
            tracer_new = np.where(
                (iadv_slev_jt <= k),
                (
                    tracer_now * rhodz_now
                    + p_dtime
                    * (
                        p_mflx_tracer_v[:, 1:] * deepatmo_divzl
                        - p_mflx_tracer_v[:, :-1] * deepatmo_divzu
                    )
                )
                / rhodz_new,
                tracer_now,
            )
        else:
            tracer_new = tracer_now

        return dict(tracer_new=tracer_new)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        tracer_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rhodz_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_tracer_v = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        deepatmo_divzl = data_alloc.random_field(grid, dims.KDim)
        deepatmo_divzu = data_alloc.random_field(grid, dims.KDim)
        rhodz_new = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        tracer_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        k = data_alloc.index_field(grid, dims.KDim)
        p_dtime = np.float64(5.0)
        ivadv_tracer = 1
        iadv_slev_jt = 4
        return dict(
            tracer_now=tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_new=rhodz_new,
            tracer_new=tracer_new,
            k=k,
            p_dtime=p_dtime,
            ivadv_tracer=ivadv_tracer,
            iadv_slev_jt=iadv_slev_jt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
