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

from icon4py.model.atmosphere.dycore.stencils.fused_velocity_advection_stencil_8_to_13 import (
    fused_velocity_advection_stencil_8_to_13,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest

from .test_copy_cell_kdim_field_to_vp import copy_cell_kdim_field_to_vp_numpy
from .test_correct_contravariant_vertical_velocity import (
    correct_contravariant_vertical_velocity_numpy,
)
from .test_init_cell_kdim_field_with_zero_vp import init_cell_kdim_field_with_zero_vp_numpy
from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy
from .test_interpolate_to_half_levels_vp import interpolate_to_half_levels_vp_numpy


class TestFusedVelocityAdvectionStencil8To13(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_8_to_13
    OUTPUTS = (
        "z_ekinh",
        "w_concorr_c",
        "z_w_con_c",
    )
    MARKERS = (pytest.mark.requires_concat_where,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        z_kin_hor_e: np.ndarray,
        e_bln_c_s: np.ndarray,
        z_w_concorr_me: np.ndarray,
        wgtfac_c: np.ndarray,
        w: np.ndarray,
        z_w_concorr_mc: np.ndarray,
        w_concorr_c: np.ndarray,
        z_ekinh: np.ndarray,
        k: np.ndarray,
        cell: np.ndarray,
        istep: int,
        nlev: ta.wpfloat,
        nflatlev: ta.wpfloat,
        z_w_con_c: np.ndarray,
        lateral_boundary_4: int,
        lateral_boundary_3: int,
        end_halo: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
    ) -> dict:
        lateral_boundary = lateral_boundary_4 if istep == 1 else lateral_boundary_3
        k_nlev = k[:-1]

        z_ekinh_cp = z_ekinh.copy()
        w_concorr_c_cp = w_concorr_c.copy()
        z_w_con_c_cp = z_w_con_c.copy()

        z_ekinh = np.where(
            k_nlev < nlev,
            interpolate_to_cell_center_numpy(connectivities, z_kin_hor_e, e_bln_c_s),
            z_ekinh,
        )

        if istep == 1:
            z_w_concorr_mc = interpolate_to_cell_center_numpy(
                connectivities, z_w_concorr_me, e_bln_c_s
            )

            w_concorr_c = np.where(
                (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
                interpolate_to_half_levels_vp_numpy(wgtfac_c=wgtfac_c, interpolant=z_w_concorr_mc),
                w_concorr_c,
            )

        z_w_con_c = np.where(
            k < nlev,
            copy_cell_kdim_field_to_vp_numpy(w),
            init_cell_kdim_field_with_zero_vp_numpy(z_w_con_c),
        )

        z_w_con_c[:, :-1] = np.where(
            (nflatlev + 1 <= k_nlev) & (k_nlev < nlev),
            correct_contravariant_vertical_velocity_numpy(z_w_con_c[:, :-1], w_concorr_c),
            z_w_con_c[:, :-1],
        )
        z_ekinh_cp[lateral_boundary:end_halo, :] = z_ekinh[lateral_boundary:end_halo, :]
        w_concorr_c_cp[lateral_boundary:end_halo, :] = w_concorr_c[lateral_boundary:end_halo, :]
        z_w_con_c_cp[lateral_boundary:end_halo, :] = z_w_con_c[lateral_boundary:end_halo, :]

        return dict(
            z_ekinh=z_ekinh_cp,
            w_concorr_c=w_concorr_c_cp,
            z_w_con_c=z_w_con_c_cp,
        )

    @pytest.fixture
    def input_data(self, grid):
        z_kin_hor_e = random_field(grid, dims.EdgeDim, dims.KDim)
        e_bln_c_s = random_field(grid, dims.CellDim, dims.C2EDim)
        z_ekinh = zero_field(grid, dims.CellDim, dims.KDim)
        z_w_concorr_me = random_field(grid, dims.EdgeDim, dims.KDim)
        z_w_concorr_mc = zero_field(grid, dims.CellDim, dims.KDim)
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim)
        w_concorr_c = zero_field(grid, dims.CellDim, dims.KDim)
        w = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        z_w_con_c = zero_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})

        k = data_alloc.allocate_indices(dims.KDim, grid=grid, is_halfdim=True)
        cell = data_alloc.allocate_indices(dims.CellDim, grid=grid)

        nlev = grid.num_levels
        nflatlev = 4

        istep = 1

        cell_domain = h_grid.domain(dims.CellDim)
        lateral_boundary_3 = (
            grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
            if hasattr(grid, "start_index")
            else 0
        )
        lateral_boundary_4 = (
            grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            if hasattr(grid, "start_index")
            else 0
        )
        end_halo = (
            grid.end_index(cell_domain(h_grid.Zone.HALO)) if hasattr(grid, "end_index") else 0
        )

        horizontal_start = 0
        horizontal_end = grid.num_cells
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            w=w,
            z_w_concorr_mc=z_w_concorr_mc,
            w_concorr_c=w_concorr_c,
            z_ekinh=z_ekinh,
            z_w_con_c=z_w_con_c,
            k=k,
            cell=cell,
            istep=istep,
            nlev=nlev,
            nflatlev=nflatlev,
            lateral_boundary_4=lateral_boundary_4,
            lateral_boundary_3=lateral_boundary_3,
            end_halo=end_halo,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
