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

import enum
import functools
import logging
import math
import pathlib

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.diffusion import diffusion_states as diffus_states
from icon4py.model.atmosphere.dycore import init_exner_pr
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import constants as phy_const, field_type_aliases as fa
from icon4py.model.common.decomposition import (
    definitions as decomposition,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.dimension import (
    CEDim,
    CellDim,
    EdgeDim,
    KDim,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils import (
    cell_2_edge_interpolation,
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
    serialbox_utils as sb,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.driver import (
    jablonowski_willamson_testcase as jw_func,
    serialbox_helpers as driver_sb,
    testcase_functions as testcase_func,
)


GRID_LEVEL = 4
GRID_ROOT = 2
GLOBAL_GRID_ID = dt_utils.GRID_IDS[dt_utils.GLOBAL_EXPERIMENT]

SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."
INITIALIZATION_ERROR_MSG = (
    "Only ANY (read from serialized data) and JABW are implemented for model initialization."
)

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(str, enum.Enum):
    SB = "serialbox"
    NC = "netcdf"


class ExperimentType(str, enum.Enum):
    JABW = "jabw"
    """initial condition of Jablonowski-Williamson test"""
    ANY = "any"
    """any test with initial conditions read from serialized data (remember to set correct SIMULATION_START_DATE)"""


def read_icon_grid(
    path: pathlib.Path,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_id=GLOBAL_GRID_ID,
    grid_root=GRID_ROOT,
    grid_level=GRID_LEVEL,
) -> icon_grid.IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        rank: mpi rank of the current compute node
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads from ppser serialized test data
        grid_id: id (uuid) of the horizontal grid
        grid_root: global grid root division number
        grid_level: global grid refinement number
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return (
            sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)
            .from_savepoint_grid(grid_id, grid_root, grid_level)
            .construct_icon_grid(on_gpu=False)
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def model_initialization_jabw(
    grid: icon_grid.IconGrid,
    cell_param: h_grid.CellParams,
    edge_param: h_grid.EdgeParams,
    path: pathlib.Path,
    rank=0,
) -> tuple[
    diffus_states.DiffusionDiagnosticState,
    solve_nh_states.DiagnosticStateNonHydro,
    solve_nh_states.PrepAdvection,
    float,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Initial condition of Jablonowski-Williamson test. Set jw_up to values larger than 0.01 if
    you want to run baroclinic case.

    Args:
        grid: IconGrid
        cell_param: cell properties
        edge_param: edge properties
        path: path where to find the input data
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """
    data_provider = sb.IconSerialDataProvider(
        "icon_pydycore", str(path.absolute()), False, mpi_rank=rank
    )

    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().asnumpy()
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().asnumpy()
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().asnumpy()
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().asnumpy()
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().asnumpy()
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().asnumpy()
    geopot = data_provider.from_metrics_savepoint().geopot().asnumpy()

    cell_lat = cell_param.cell_center_lat.asnumpy()
    edge_lat = edge_param.edge_center[0].asnumpy()
    edge_lon = edge_param.edge_center[1].asnumpy()
    primal_normal_x = edge_param.primal_normal[0].asnumpy()

    cell_2_edge_coeff = data_provider.from_interpolation_savepoint().c_lin_e()
    rbf_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbf_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()

    cell_size = grid.num_cells
    num_levels = grid.num_levels

    grid_idx_edge_start_plus1 = grid.get_end_index(
        EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1
    )
    grid_idx_edge_end = grid.get_end_index(EdgeDim, h_grid.HorizontalMarkerIndex.end(EdgeDim))
    grid_idx_cell_interior_start = grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.interior(CellDim)
    )
    grid_idx_cell_start_plus1 = grid.get_end_index(
        CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
    )
    grid_idx_cell_end = grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.end(CellDim))

    p_sfc = 100000.0
    jw_up = 0.0  # if doing baroclinic wave test, please set it to a nonzero value
    jw_u0 = 35.0
    jw_temp0 = 288.0
    # DEFINED PARAMETERS for jablonowski williamson:
    eta_0 = 0.252
    eta_t = 0.2  # tropopause
    gamma = 0.005  # temperature elapse rate (K/m)
    dtemp = 4.8e5  # empirical temperature difference (K)
    # for baroclinic wave test
    lon_perturbation_center = math.pi / 9.0  # longitude of the perturb centre
    lat_perturbation_center = 2.0 * lon_perturbation_center  # latitude of the perturb centre
    ps_o_p0ref = p_sfc / phy_const.P0REF

    w_numpy = np.zeros((cell_size, num_levels + 1), dtype=float)
    exner_numpy = np.zeros((cell_size, num_levels), dtype=float)
    rho_numpy = np.zeros((cell_size, num_levels), dtype=float)
    temperature_numpy = np.zeros((cell_size, num_levels), dtype=float)
    pressure_numpy = np.zeros((cell_size, num_levels), dtype=float)
    theta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)
    eta_v_numpy = np.zeros((cell_size, num_levels), dtype=float)

    sin_lat = np.sin(cell_lat)
    cos_lat = np.cos(cell_lat)
    fac1 = 1.0 / 6.3 - 2.0 * (sin_lat**6) * (cos_lat**2 + 1.0 / 3.0)
    fac2 = (
        (8.0 / 5.0 * (cos_lat**3) * (sin_lat**2 + 2.0 / 3.0) - 0.25 * math.pi)
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    lapse_rate = phy_const.RD * gamma / phy_const.GRAV
    for k_index in range(num_levels - 1, -1, -1):
        eta_old = np.full(cell_size, fill_value=1.0e-7, dtype=float)
        log.info(f"In Newton iteration, k = {k_index}")
        # Newton iteration to determine zeta
        for _ in range(100):
            eta_v_numpy[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = np.cos(eta_v_numpy[:, k_index])
            sin_etav = np.sin(eta_v_numpy[:, k_index])

            temperature_avg = jw_temp0 * (eta_old**lapse_rate)
            geopot_avg = jw_temp0 * phy_const.GRAV / gamma * (1.0 - eta_old**lapse_rate)
            temperature_avg = np.where(
                eta_old < eta_t, temperature_avg + dtemp * ((eta_t - eta_old) ** 5), temperature_avg
            )
            geopot_avg = np.where(
                eta_old < eta_t,
                geopot_avg
                - phy_const.RD
                * dtemp
                * (
                    (np.log(eta_old / eta_t) + 137.0 / 60.0) * (eta_t**5)
                    - 5.0 * (eta_t**4) * eta_old
                    + 5.0 * (eta_t**3) * (eta_old**2)
                    - 10.0 / 3.0 * (eta_t**2) * (eta_old**3)
                    + 1.25 * eta_t * (eta_old**4)
                    - 0.2 * (eta_old**5)
                ),
                geopot_avg,
            )

            geopot_jw = geopot_avg + jw_u0 * (cos_etav**1.5) * (
                fac1 * jw_u0 * (cos_etav**1.5) + fac2
            )
            temperature_jw = (
                temperature_avg
                + 0.75
                * eta_old
                * math.pi
                * jw_u0
                / phy_const.RD
                * sin_etav
                * np.sqrt(cos_etav)
                * (2.0 * jw_u0 * fac1 * (cos_etav**1.5) + fac2)
            )
            newton_function = geopot_jw - geopot[:, k_index]
            newton_function_prime = -phy_const.RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        # Final update for zeta_v
        eta_v_numpy[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
        # Use analytic expressions at all model level
        exner_numpy[:, k_index] = (eta_old * ps_o_p0ref) ** phy_const.RD_O_CPD
        theta_v_numpy[:, k_index] = temperature_jw / exner_numpy[:, k_index]
        rho_numpy[:, k_index] = (
            exner_numpy[:, k_index] ** phy_const.CVD_O_RD
            * phy_const.P0REF
            / phy_const.RD
            / theta_v_numpy[:, k_index]
        )
        # initialize diagnose pressure and temperature variables
        pressure_numpy[:, k_index] = phy_const.P0REF * exner_numpy[:, k_index] ** phy_const.CPD_O_RD
        temperature_numpy[:, k_index] = temperature_jw
    log.info("Newton iteration completed!")

    eta_v = gtx.as_field((CellDim, KDim), eta_v_numpy)
    eta_v_e = field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid)
    cell_2_edge_interpolation.cell_2_edge_interpolation(
        eta_v,
        cell_2_edge_coeff,
        eta_v_e,
        grid_idx_edge_start_plus1,
        grid_idx_edge_end,
        0,
        num_levels,
        offset_provider=grid.offset_providers,
    )
    log.info("Cell-to-edge eta_v computation completed.")

    vn_numpy = jw_func.zonalwind_2_normalwind_jabw_numpy(
        grid,
        jw_u0,
        jw_up,
        lat_perturbation_center,
        lon_perturbation_center,
        edge_lat,
        edge_lon,
        primal_normal_x,
        eta_v_e.asnumpy(),
    )
    log.info("U2vn computation completed.")

    rho_numpy, exner_numpy, theta_v_numpy = testcase_func.hydrostatic_adjustment_numpy(
        wgtfac_c,
        ddqz_z_half,
        exner_ref_mc,
        d_exner_dz_ref_ic,
        theta_ref_mc,
        theta_ref_ic,
        rho_numpy,
        exner_numpy,
        theta_v_numpy,
        num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    vn = gtx.as_field((EdgeDim, KDim), vn_numpy)
    w = gtx.as_field((CellDim, KDim), w_numpy)
    exner = gtx.as_field((CellDim, KDim), exner_numpy)
    rho = gtx.as_field((CellDim, KDim), rho_numpy)
    temperature = gtx.as_field((CellDim, KDim), temperature_numpy)
    pressure = gtx.as_field((CellDim, KDim), pressure_numpy)
    theta_v = gtx.as_field((CellDim, KDim), theta_v_numpy)
    pressure_ifc_numpy = np.zeros((cell_size, num_levels + 1), dtype=float)
    pressure_ifc_numpy[:, -1] = p_sfc
    pressure_ifc = gtx.as_field((CellDim, KDim), pressure_ifc_numpy)

    vn_next = gtx.as_field((EdgeDim, KDim), vn_numpy)
    w_next = gtx.as_field((CellDim, KDim), w_numpy)
    exner_next = gtx.as_field((CellDim, KDim), exner_numpy)
    rho_next = gtx.as_field((CellDim, KDim), rho_numpy)
    theta_v_next = gtx.as_field((CellDim, KDim), theta_v_numpy)

    u = field_alloc.allocate_zero_field(CellDim, KDim, grid=grid)
    v = field_alloc.allocate_zero_field(CellDim, KDim, grid=grid)
    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation(
        vn,
        rbf_vec_coeff_c1,
        rbf_vec_coeff_c2,
        u,
        v,
        grid_idx_cell_start_plus1,
        grid_idx_cell_end,
        0,
        num_levels,
        offset_provider=grid.offset_providers,
    )

    log.info("U, V computation completed.")

    exner_pr = field_alloc.allocate_zero_field(CellDim, KDim, grid=grid)
    init_exner_pr.init_exner_pr(
        exner,
        data_provider.from_metrics_savepoint().exner_ref_mc(),
        exner_pr,
        grid_idx_cell_interior_start,
        grid_idx_cell_end,
        0,
        grid.num_levels,
        offset_provider={},
    )
    log.info("exner_pr initialization completed.")

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        u=u,
        v=v,
    )

    prognostic_state_now = prognostics.PrognosticState(
        w=w,
        vn=vn,
        theta_v=theta_v,
        rho=rho,
        exner=exner,
    )
    prognostic_state_next = prognostics.PrognosticState(
        w=w_next,
        vn=vn_next,
        theta_v=theta_v_next,
        rho=rho_next,
        exner=exner_next,
    )

    diffusion_diagnostic_state = diffus_states.DiffusionDiagnosticState(
        hdef_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        div_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        dwdx=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        dwdy=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
    )
    solve_nonhydro_diagnostic_state = solve_nh_states.DiagnosticStateNonHydro(
        theta_v_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        exner_pr=exner_pr,
        rho_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        ddt_exner_phy=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        grf_tend_rho=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        grf_tend_thv=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        grf_tend_w=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        mass_fl_e=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        ddt_vn_phy=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        grf_tend_vn=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        ddt_vn_apc_ntl1=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        ddt_vn_apc_ntl2=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        ddt_w_adv_ntl1=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        ddt_w_adv_ntl2=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        vt=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        vn_ie=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid, is_halfdim=True),
        w_concorr_c=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
    )

    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        mass_flx_me=field_alloc.allocate_zero_field(EdgeDim, KDim, grid=grid),
        mass_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
    )
    log.info("Initialization completed.")

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        0.0,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def model_initialization_serialbox(
    grid: icon_grid.IconGrid, path: pathlib.Path, rank=0
) -> tuple[
    diffus_states.DiffusionDiagnosticState,
    solve_nh_states.DiagnosticStateNonHydro,
    solve_nh_states.PrepAdvection,
    float,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Initial condition read from serialized data. Diagnostic variables are allocated as zero
    fields.

    Args:
        grid: IconGrid
        path: path where to find the input data
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """

    data_provider = _serial_data_provider(path, rank)
    diffusion_init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    solve_nonhydro_init_savepoint = data_provider.from_savepoint_nonhydro_init(
        istep=1, date=SIMULATION_START_DATE, jstep=0
    )
    velocity_init_savepoint = data_provider.from_savepoint_velocity_init(
        istep=1, vn_only=False, date=SIMULATION_START_DATE, jstep=0
    )
    prognostic_state_now = diffusion_init_savepoint.construct_prognostics()
    diffusion_diagnostic_state = driver_sb.construct_diagnostics_for_diffusion(
        diffusion_init_savepoint,
    )
    solve_nonhydro_diagnostic_state = solve_nh_states.DiagnosticStateNonHydro(
        theta_v_ic=solve_nonhydro_init_savepoint.theta_v_ic(),
        exner_pr=solve_nonhydro_init_savepoint.exner_pr(),
        rho_ic=solve_nonhydro_init_savepoint.rho_ic(),
        ddt_exner_phy=solve_nonhydro_init_savepoint.ddt_exner_phy(),
        grf_tend_rho=solve_nonhydro_init_savepoint.grf_tend_rho(),
        grf_tend_thv=solve_nonhydro_init_savepoint.grf_tend_thv(),
        grf_tend_w=solve_nonhydro_init_savepoint.grf_tend_w(),
        mass_fl_e=solve_nonhydro_init_savepoint.mass_fl_e(),
        ddt_vn_phy=solve_nonhydro_init_savepoint.ddt_vn_phy(),
        grf_tend_vn=solve_nonhydro_init_savepoint.grf_tend_vn(),
        ddt_vn_apc_ntl1=velocity_init_savepoint.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=velocity_init_savepoint.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=velocity_init_savepoint.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=velocity_init_savepoint.ddt_w_adv_pc(2),
        vt=velocity_init_savepoint.vt(),
        vn_ie=velocity_init_savepoint.vn_ie(),
        w_concorr_c=velocity_init_savepoint.w_concorr_c(),
        rho_incr=None,  # solve_nonhydro_init_savepoint.rho_incr(),
        vn_incr=None,  # solve_nonhydro_init_savepoint.vn_incr(),
        exner_incr=None,  # solve_nonhydro_init_savepoint.exner_incr(),
        exner_dyn_incr=None,
    )

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        pressure_ifc=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid, is_halfdim=True),
        temperature=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        u=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
        v=field_alloc.allocate_zero_field(CellDim, KDim, grid=grid),
    )

    prognostic_state_next = prognostics.PrognosticState(
        w=solve_nonhydro_init_savepoint.w_new(),
        vn=solve_nonhydro_init_savepoint.vn_new(),
        theta_v=solve_nonhydro_init_savepoint.theta_v_new(),
        rho=solve_nonhydro_init_savepoint.rho_new(),
        exner=solve_nonhydro_init_savepoint.exner_new(),
    )

    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=solve_nonhydro_init_savepoint.vn_traj(),
        mass_flx_me=solve_nonhydro_init_savepoint.mass_flx_me(),
        mass_flx_ic=solve_nonhydro_init_savepoint.mass_flx_ic(),
    )

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        solve_nonhydro_init_savepoint.divdamp_fac_o2(),
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_initial_state(
    grid: icon_grid.IconGrid,
    cell_param: h_grid.CellParams,
    edge_param: h_grid.EdgeParams,
    path: pathlib.Path,
    rank=0,
    experiment_type: ExperimentType = ExperimentType.ANY,
) -> tuple[
    diffus_states.DiffusionDiagnosticState,
    solve_nh_states.DiagnosticStateNonHydro,
    solve_nh_states.PrepAdvection,
    float,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Read initial prognostic and diagnostic fields.

    Args:
        grid: IconGrid
        cell_param: cell properties
        edge_param: edge properties
        path: path to the serialized input data
        rank: mpi rank of the current compute node
        experiment_type: (optional) defaults to ANY=any, type of initial condition to be read

    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """
    if experiment_type == ExperimentType.JABW:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_jabw(grid, cell_param, edge_param, path, rank)
    elif experiment_type == ExperimentType.ANY:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_serialbox(grid, path, rank)
    else:
        raise NotImplementedError(INITIALIZATION_ERROR_MSG)

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_geometry_fields(
    path: pathlib.Path,
    vertical_grid_config: v_grid.VerticalGridConfig,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
    grid_id=GLOBAL_GRID_ID,
    grid_root=GRID_ROOT,
    grid_level=GRID_LEVEL,
) -> tuple[h_grid.EdgeParams, h_grid.CellParams, v_grid.VerticalGridParams, fa.CellField[bool]]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        vertical_grid_config: Vertical grid configuration
        rank: mpi rank of the current compute node
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read
        grid_id: id (uuid) of the horizontal grid
        grid_root: global grid root division number
        grid_level: global grid refinement number

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = _grid_savepoint(path, rank, grid_id, grid_root, grid_level)
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_grid_config)
        vertical_geometry = v_grid.VerticalGridParams(
            vertical_config=vertical_grid_config,
            vct_a=vct_a,
            vct_b=vct_b,
            _min_index_flat_horizontal_grad_pressure=sp.nflat_gradp(),
        )
        return edge_geometry, cell_geometry, vertical_geometry, sp.c_owner_mask()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


@functools.cache
def _serial_data_provider(path, rank) -> sb.IconSerialDataProvider:
    return sb.IconSerialDataProvider("icon_pydycore", str(path.absolute()), False, mpi_rank=rank)


@functools.cache
def _grid_savepoint(path, rank, grid_id, grid_root, grid_level) -> sb.IconGridSavepoint:
    sp = _serial_data_provider(path, rank).from_savepoint_grid(grid_id, grid_root, grid_level)
    return sp


def read_decomp_info(
    path: pathlib.Path,
    procs_props: decomposition.ProcessProperties,
    ser_type=SerializationType.SB,
    grid_id=GLOBAL_GRID_ID,
    grid_root=GRID_ROOT,
    grid_level=GRID_LEVEL,
) -> decomposition.DecompositionInfo:
    if ser_type == SerializationType.SB:
        return _grid_savepoint(
            path, procs_props.rank, grid_id, grid_root, grid_level
        ).construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    grid: icon_grid.IconGrid,
    path: pathlib.Path,
    rank=0,
    ser_type: SerializationType = SerializationType.SB,
) -> tuple[
    diffus_states.DiffusionMetricState,
    diffus_states.DiffusionInterpolationState,
    solve_nh_states.MetricStateNonHydro,
    solve_nh_states.InterpolationState,
    diagnostics.DiagnosticMetricState,
]:
    """
    Read fields for metric and interpolation state.

     Args:
        grid: IconGrid
        path: path to the serialized input data
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        data_provider = _serial_data_provider(path, rank)

        diffusion_interpolation_state = driver_sb.construct_interpolation_state_for_diffusion(
            data_provider.from_interpolation_savepoint()
        )
        diffusion_metric_state = driver_sb.construct_metric_state_for_diffusion(
            data_provider.from_metrics_savepoint()
        )
        interpolation_savepoint = data_provider.from_interpolation_savepoint()
        grg = interpolation_savepoint.geofac_grg()
        solve_nonhydro_interpolation_state = solve_nh_states.InterpolationState(
            c_lin_e=interpolation_savepoint.c_lin_e(),
            c_intp=interpolation_savepoint.c_intp(),
            e_flx_avg=interpolation_savepoint.e_flx_avg(),
            geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
            geofac_rot=interpolation_savepoint.geofac_rot(),
            pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), CEDim),
            rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
            geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), CEDim),
            geofac_n2s=interpolation_savepoint.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        solve_nonhydro_metric_state = solve_nh_states.MetricStateNonHydro(
            bdy_halo_c=metrics_savepoint.bdy_halo_c(),
            mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
            rayleigh_w=metrics_savepoint.rayleigh_w(),
            exner_exfac=metrics_savepoint.exner_exfac(),
            exner_ref_mc=metrics_savepoint.exner_ref_mc(),
            wgtfac_c=metrics_savepoint.wgtfac_c(),
            wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
            inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
            rho_ref_mc=metrics_savepoint.rho_ref_mc(),
            theta_ref_mc=metrics_savepoint.theta_ref_mc(),
            vwind_expl_wgt=metrics_savepoint.vwind_expl_wgt(),
            d_exner_dz_ref_ic=metrics_savepoint.d_exner_dz_ref_ic(),
            ddqz_z_half=metrics_savepoint.ddqz_z_half(),
            theta_ref_ic=metrics_savepoint.theta_ref_ic(),
            d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
            d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
            rho_ref_me=metrics_savepoint.rho_ref_me(),
            theta_ref_me=metrics_savepoint.theta_ref_me(),
            ddxn_z_full=metrics_savepoint.ddxn_z_full(),
            zdiff_gradp=metrics_savepoint.zdiff_gradp(),
            vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
            ipeidx_dsl=metrics_savepoint.ipeidx_dsl(),
            pg_exdist=metrics_savepoint.pg_exdist(),
            ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
            ddxt_z_full=metrics_savepoint.ddxt_z_full(),
            wgtfac_e=metrics_savepoint.wgtfac_e(),
            wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(grid.num_levels),
            vwind_impl_wgt=metrics_savepoint.vwind_impl_wgt(),
            hmask_dd3d=metrics_savepoint.hmask_dd3d(),
            scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
            coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
            coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
            coeff_gradekin=metrics_savepoint.coeff_gradekin(),
        )

        diagnostic_metric_state = diagnostics.DiagnosticMetricState(
            ddqz_z_full=metrics_savepoint.ddqz_z_full(),
            rbf_vec_coeff_c1=interpolation_savepoint.rbf_vec_coeff_c1(),
            rbf_vec_coeff_c2=interpolation_savepoint.rbf_vec_coeff_c2(),
        )

        return (
            diffusion_metric_state,
            diffusion_interpolation_state,
            solve_nonhydro_metric_state,
            solve_nonhydro_interpolation_state,
            diagnostic_metric_state,
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(
    run_path: str, experiment_name: str, processor_procs: decomposition.ProcessProperties = None
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation

    """
    run_dir = (
        pathlib.Path(run_path).absolute() if run_path else pathlib.Path(__file__).absolute().parent
    )
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(f"dummy_dycore_driver_{experiment_name}.log")
    logfile.touch(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    # TODO (Chia Rui): modify here when single_dispatch is ready
    console_handler.addFilter(mpi_decomp.ParallelLogger(processor_procs))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)
