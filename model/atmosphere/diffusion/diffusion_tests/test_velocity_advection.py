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

from atm_dyn_iconam.tests.test_utils.helpers import as_1D_sparse_field, dallclose
from icon4py.model.common.dimension import CEDim
from icon4py.grid.horizontal import CellParams, EdgeParams
from icon4py.grid.vertical import VerticalModelParams
from icon4py.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.velocity.velocity_advection import VelocityAdvection


@pytest.mark.datatest
def test_velocity_init(
    savepoint_velocity_init,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    icon_grid,
    step_date_init,
    damping_height,
):

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height, nflat_gradp=0, nflatlev=0
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    assert dallclose(0.0, np.asarray(velocity_advection.cfl_clipping))
    assert dallclose(0.0, np.asarray(velocity_advection.pre_levelmask))
    assert dallclose(False, np.asarray(velocity_advection.levelmask))
    assert dallclose(0.0, np.asarray(velocity_advection.vcfl))

    assert velocity_advection.cfl_w_limit == 0.65
    assert velocity_advection.scalfac_exdiff == 0.05


@pytest.mark.datatest
def test_verify_velocity_init_against_regular_savepoint(
    savepoint_velocity_init,
    interpolation_savepoint,
    grid_savepoint,
    metrics_savepoint,
    icon_grid,
    damping_height,
):
    savepoint = savepoint_velocity_init
    dtime = savepoint.get_metadata("dtime").get("dtime")

    interpolation_state = interpolation_savepoint.construct_interpolation_state()
    metric_state = metrics_savepoint.construct_metric_state()
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(), rayleigh_damping_height=damping_height, nflatlev=0, nflat_gradp=0
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )
    velocity_advection.init(
        grid=icon_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    assert savepoint.cfl_w_limit() == velocity_advection.cfl_w_limit / dtime
    assert savepoint.scalfac_exdiff() == velocity_advection.scalfac_exdiff / (
        dtime * (0.85 - savepoint.cfl_w_limit() * dtime)
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(1, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_velocity_predictor_step(
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    data_provider,
    metrics_nonhydro_savepoint,
    interpolation_savepoint,
    savepoint_velocity_exit,
):
    sp_v = savepoint_velocity_init
    sp_d = data_provider.from_savepoint_grid()
    sp_int = interpolation_savepoint
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")
    scalfac_exdiff = sp_v.scalfac_exdiff()

    diagnostic_state = DiagnosticStateNonHydro(
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )
    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=None,
        rho=None,
        exner=None,
    )
    interpolation_state = interpolation_savepoint.construct_interpolation_state()

    metric_state_nonhydro = metrics_nonhydro_savepoint.construct_nh_metric_state(icon_grid.n_lev())

    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=int32(grid_savepoint.nflatlev()),
        nflat_gradp=int32(grid_savepoint.nflat_gradp()),
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    velocity_advection.run_predictor_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_w_concorr_me=sp_v.z_w_concorr_me(),
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        dtime=dtime,
        ntnd=ntnd - 1,
        tangent_orientation=edge_geometry.tangent_orientation,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.c_owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(ntnd)
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(ntnd)
    # icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()
    icon_result_vn_ie = savepoint_velocity_exit.vn_ie()
    icon_result_vt = savepoint_velocity_exit.vt()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c()
    icon_result_z_w_concorr_mc = savepoint_velocity_exit.z_w_concorr_mc()
    icon_result_z_vt_ie = savepoint_velocity_exit.z_vt_ie()

    icon_result_z_kin_hor_e = savepoint_velocity_exit.z_kin_hor_e()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w()
    icon_result_z_w_con_c_full = savepoint_velocity_exit.z_w_con_c_full()
    icon_result_z_w_concorr_me = savepoint_velocity_exit.z_w_concorr_me()
    #   assert dallclose(np.asarray(icon_result_z_v_grad_w), np.asarray(diagnostic_state.z_v_grad_w))

    # stencil 01
    assert dallclose(np.asarray(icon_result_vt), np.asarray(diagnostic_state.vt), atol=1.0e-14)
    # stencil 02,05
    assert dallclose(
        np.asarray(icon_result_vn_ie), np.asarray(diagnostic_state.vn_ie), atol=1.0e-14
    )
    # stencil 02
    #assert dallclose(np.asarray(icon_result_z_kin_hor_e)[2538:31558, :],
    #            np.asarray(velocity_advection.z_kin_hor_e)[2538:31558, :])
    # stencil 03,05,06
    #assert dallclose(np.asarray(icon_result_z_vt_ie), np.asarray(velocity_advection.z_vt_ie))
    # stencil 04
    #assert dallclose(
    #    np.asarray(icon_result_z_w_concorr_me)[:,vertical_params.nflatlev:icon_grid.n_lev()], np.asarray(velocity_advection.z_w_concorr_me)[:,vertical_params.nflatlev:icon_grid.n_lev()]
    #)
    # stencil 07
    assert dallclose(np.asarray(icon_result_z_v_grad_w)[3777:31558, :],
                       np.asarray(velocity_advection.z_v_grad_w)[3777:31558, :], atol=1.0e-16)
    # stencil 08
    assert dallclose(np.asarray(savepoint_velocity_exit.z_ekinh())[3316:20896, :],
                np.asarray(velocity_advection.z_ekinh)[3316:20896, :])
    # stencil 09
    assert dallclose(
        np.asarray(icon_result_z_w_concorr_mc)[3316:20896,vertical_params.nflatlev:icon_grid.n_lev()], np.asarray(velocity_advection.z_w_concorr_mc)[3316:20896,vertical_params.nflatlev:icon_grid.n_lev()],
        atol=1.0e-15
    )
    # stencil 10
    assert dallclose(
        np.asarray(icon_result_w_concorr_c)[3316:20896, vertical_params.nflatlev + 1:icon_grid.n_lev()],
        np.asarray(diagnostic_state.w_concorr_c)[3316:20896, vertical_params.nflatlev + 1:icon_grid.n_lev()],
        atol=1.0e-15
    )
    # stencil 11,12,13,14
    assert dallclose(np.asarray(savepoint_velocity_exit.z_w_con_c())[3316:20896,:],
                     np.asarray(velocity_advection.z_w_con_c)[3316:20896,:],
                     atol=1.0e-15)
    #stencil 16
    assert dallclose(
        np.asarray(icon_result_ddt_w_adv_pc)[3316:20896,:],
        np.asarray(diagnostic_state.ddt_w_adv_pc[ntnd-1])[3316:20896,:],
        atol=5.0e-16, rtol=1.0e-10
    )
    # stencil 19 level 0 not verifying
    assert dallclose(
        np.asarray(icon_result_ddt_vn_apc_pc)[5387:31558,0:65],
        np.asarray(diagnostic_state.ddt_vn_apc_pc[ntnd-1])[5387:31558,0:65],
        atol=1.0e-15,
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep, step_date_init, step_date_exit",
    [(2, "2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")],
)
def test_velocity_corrector_step(
    damping_height,
    icon_grid,
    grid_savepoint,
    savepoint_velocity_init,
    data_provider,
    savepoint_velocity_exit,
    interpolation_savepoint,
    metrics_savepoint,
    metrics_nonhydro_savepoint
):
    sp_v = savepoint_velocity_init
    sp_d = data_provider.from_savepoint_grid()
    sp_int = interpolation_savepoint
    sp_met = metrics_savepoint
    vn_only = sp_v.get_metadata("vn_only").get("vn_only")
    ntnd = sp_v.get_metadata("ntnd").get("ntnd")
    dtime = sp_v.get_metadata("dtime").get("dtime")
    cfl_w_limit = sp_v.cfl_w_limit()
    scalfac_exdiff = sp_v.scalfac_exdiff()

    diagnostic_state = DiagnosticStateNonHydro(
        vt=sp_v.vt(),
        vn_ie=sp_v.vn_ie(),
        w_concorr_c=sp_v.w_concorr_c(),
        theta_v_ic=None,
        exner_pr=None,
        rho_ic=None,
        ddt_exner_phy=None,
        grf_tend_rho=None,
        grf_tend_thv=None,
        grf_tend_w=None,
        mass_fl_e=None,
        ddt_vn_phy=None,
        grf_tend_vn=None,
        ddt_vn_apc_ntl1=sp_v.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=sp_v.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=sp_v.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=sp_v.ddt_w_adv_pc(2),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
    )
    prognostic_state = PrognosticState(
        w=sp_v.w(),
        vn=sp_v.vn(),
        exner_pressure=None,
        theta_v=None,
        rho=None,
        exner=None,
    )

    interpolation_state = interpolation_savepoint.construct_interpolation_state()

    metric_state_nonhydro = metrics_nonhydro_savepoint.construct_nh_metric_state(icon_grid.n_lev())


    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()

    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev= int32(grid_savepoint.nflatlev()),
        nflat_gradp=int32(grid_savepoint.nflat_gradp()),
    )

    velocity_advection = VelocityAdvection(
        grid=icon_grid,
        metric_state=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
    )

    velocity_advection.run_corrector_step(
        vn_only=vn_only,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        z_kin_hor_e=sp_v.z_kin_hor_e(),
        z_vt_ie=sp_v.z_vt_ie(),
        inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
        dtime=dtime,
        ntnd=ntnd - 1,
        tangent_orientation=edge_geometry.tangent_orientation,
        cfl_w_limit=sp_v.cfl_w_limit(),
        scalfac_exdiff=scalfac_exdiff,
        cell_areas=cell_geometry.area,
        owner_mask=sp_d.c_owner_mask(),
        f_e=sp_d.f_e(),
        area_edge=edge_geometry.edge_areas,
    )

    icon_result_ddt_vn_apc_pc = savepoint_velocity_exit.ddt_vn_apc_pc(ntnd)
    icon_result_ddt_w_adv_pc = savepoint_velocity_exit.ddt_w_adv_pc(ntnd)
    # icon_result_max_vcfl_dyn = savepoint_velocity_exit.max_vcfl_dyn()
    icon_result_w_concorr_c = savepoint_velocity_exit.w_concorr_c()
    icon_result_z_w_concorr_mc = savepoint_velocity_exit.z_w_concorr_mc()
    icon_result_z_vt_ie = savepoint_velocity_exit.z_vt_ie()

    icon_result_z_kin_hor_e = savepoint_velocity_exit.z_kin_hor_e()
    icon_result_z_v_grad_w = savepoint_velocity_exit.z_v_grad_w()
    icon_result_z_w_con_c_full = savepoint_velocity_exit.z_w_con_c_full()
    icon_result_z_w_concorr_me = savepoint_velocity_exit.z_w_concorr_me()
    #   assert dallclose(np.asarray(icon_result_z_v_grad_w), np.asarray(diagnostic_state.z_v_grad_w))

    # stencil 07
    assert dallclose(
        np.asarray(icon_result_z_v_grad_w)[3777:31558, :],
        np.asarray(velocity_advection.z_v_grad_w)[3777:31558, :],
        atol=1e-16,
    )
    # stencil 08
    assert dallclose(
        np.asarray(savepoint_velocity_exit.z_ekinh())[3316:20896, :],
        np.asarray(velocity_advection.z_ekinh)[3316:20896, :],
    )

    # stencil 11,12,13,14
    assert dallclose(np.asarray(savepoint_velocity_exit.z_w_con_c())[3316:20896, :],
                       np.asarray(velocity_advection.z_w_con_c)[3316:20896, :])
    # stencil 16
    assert dallclose(
        np.asarray(icon_result_ddt_w_adv_pc)[3316:20896, :],
        np.asarray(diagnostic_state.ddt_w_adv_pc[ntnd-1])[3316:20896, :],
        atol=5.0e-16
    )
    # stencil 19 level 0 not verifying
    assert dallclose(
        np.asarray(icon_result_ddt_vn_apc_pc)[5387:31558, 0:65],
        np.asarray(diagnostic_state.ddt_vn_apc_pc[ntnd-1])[5387:31558, 0:65],
        atol=5.0e-16
    )

