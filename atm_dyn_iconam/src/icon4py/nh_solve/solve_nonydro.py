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

from gt4py.next.ffront.fbuiltins import Fields
from gt4py.next.iterator.type_inference import Tuple

import icon4py.common.constants as constants
import icon4py.nh_solve.solve_nonhydro_program as nhsolve_prog
from icon4py.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2CDim,
    ECDim,
    ECVDim,
    EdgeDim,
    KDim,
)
from icon4py.state_utils.diagnostic_state import DiagnosticState
from icon4py.state_utils.horizontal import HorizontalMarkerIndex
from icon4py.state_utils.icon_grid import IconGrid, VerticalModelParams
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricState
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import _allocate, set_zero_c_k
from icon4py.state_utils.z_fields import ZFields
from icon4py.velocity import velocity_advection


class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    TODO: @abishekg7 to be read from config
    TODO: @abishekg7 handle dependencies on other namelists (see below...)
    """

    def __init__(
        self,
        itime_scheme: int = 5,
        iadv_rhotheta: int = 5,
        igradp_method: int = 5,
        ndyn_substeps_var: float = 3.0,
        rayleigh_type: int = 2,
        divdamp_order: int = 24,
        idiv_method: int = 4,  # TODO: @nfarabullini check if this param is ok here, in FORTRAN is imported from somewhere else
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 0.0,
        divdamp_type: int = 3,
        lhdiff_rcf: bool = True,
    ):

        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme
        self.iadv_rhotheta: int = iadv_rhotheta

        self._validate()
        self.l_open_ubc: bool
        self.igradp_method: int = igradp_method
        self.ndyn_substeps_var = ndyn_substeps_var
        self.idiv_method: int = idiv_method
        self.is_iau_active: bool = is_iau_active
        self.iau_wgt_dyn: float = iau_wgt_dyn
        self.divdamp_type: int = divdamp_type
        self.lhdiff_rcf: bool = lhdiff_rcf
        self.rayleigh_type: int = rayleigh_type
        self.divdamp_order: int = divdamp_order

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""

        if self.l_open_ubc:
            raise NotImplementedError(
                "Open upper boundary conditions not supported"
                "(to allow vertical motions related to diabatic heating to extend beyond the model top)"
            )

        if self.lvert_nest or self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method == 4 or self.igradp_method == 5:
            raise NotImplementedError("igradp_method 4 and 5 not implemented")


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):

        # start level for 3D divergence damping terms
        self.kstart_dd3d: int = (
           #TODO: @abishekg7 See mo_vertical_grid.f90
        )
        # start level for moist physics processes (specified by htop_moist_proc)
        self.kstart_moist: int # see mo_nonhydrostatic_config.f90

        self.alin = (config.divdamp_fac2 - config.divdamp_fac) / (config.divdamp_z2 - config.divdamp_z)

        self.df32 = config.divdamp_fac3 - config.divdamp_fac2
        self.dz32 = config.divdamp_z3 - config.divdamp_z2
        self.df42 = config.divdamp_fac4 - config.divdamp_fac2
        self.dz42 = config.divdamp_z4 - config.divdamp_z2

        self.bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
        self.aqdr = df32 / dz32 - bqdr * dz32


class SolveNonhydro:
    # def __init__(self, run_program=True):

    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state: MetricState,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """
        self.config: NonHydrostaticConfig = config
        self.params: NonHydrostaticParams = params
        self.grid = grid
        self.vertical_params = vertical_params
        self.metric_state: MetricState = metric_state
        self.interpolation_state: InterpolationState = interpolation_state

        self._allocate_local_fields()
        self._initialized = True
        self.rd_o_cvd = constants.RD / constants.CPD
        self.rd_o_p0ref = constants.RD / constants.P0REF

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_exner_ex_pr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_exner_ic = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_dexner_dz_c_1 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_theta_v_pr_ic = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_th_ddz_exner_c = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rth_pr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rth_pr_1 = self.z_rth_pr[
            :, 0
        ]  # TODO: @nfarabullini check if this is correct
        self.z_rth_pr_2 = self.z_rth_pr[
            :, 1
        ]  # TODO: @nfarabullini check if this is correct
        self.z_grad_rth = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_1 = self.z_rth_pr[
            :, 0
        ]  # TODO: @nfarabullini check if this is correct
        self.z_grad_rth_2 = self.z_rth_pr[
            :, 1
        ]  # TODO: @nfarabullini check if this is correct
        self.z_grad_rth_3 = self.z_rth_pr[
            :, 2
        ]  # TODO: @nfarabullini check if this is correct
        self.z_grad_rth_4 = self.z_rth_pr[
            :, 3
        ]  # TODO: @nfarabullini check if this is correct
        self.z_dexner_dz_c_2 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_gradh_exner = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_alpha = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_beta = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_w_expl = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_exner_expl = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_q = _allocate(CellDim, KDim, mesh=self.grid)
        self.exner_dyn_incr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_contr_w_fl_l = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rho_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_theta_v_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_hydro_corr = _allocate(EdgeDim, mesh=self.grid)
        self.z_vn_avg = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_theta_v_fl_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_flxdiv_mass = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_flxdiv_theta = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rho_expl = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_dwdz_dd = _allocate(CellDim, KDim, mesh=self.grid)

    # def initial_step(self):

    def time_step(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        dtime: float,
        tangent_orientation: Field[[EdgeDim], float],
        inverse_primal_edge_lengths: Field[[EdgeDim], float],
        inverse_dual_edge_length: Field[[EdgeDim], float],
        inverse_vert_vert_lengths: Field[[EdgeDim], float],
        primal_normal_vert: Tuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        dual_normal_vert: Tuple[Field[[ECVDim], float], Field[[ECVDim], float]],
        edge_areas: Field[[EdgeDim], float],
        cell_areas: Field[[CellDim], float],
    ):
        """
        Do one diffusion step within regular time loop.

        runs a diffusion step for the parameter linit=False, within regular time loop.
        """

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
        ) = self.init_dimensions_boundaries()
        # velocity_advection is referenced inside

        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0/config.ndyn_substeps_var

        #  Precompute Rayleigh damping factor
        z_raylfac = 1.0 / (
            1.0 + dtime * self.metric_state.rayleigh_w
        )  # TODO: @nfarabullini make this a program

        # Coefficient for reduced fourth-order divergence damping along nest boundaries
        bdy_divdamp = 0.75/(nudge_max_coeff + dbl_eps)*abs(scal_divdamp)

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        scal_divdamp_o2 = divdamp_fac_o2 * p_patch%geometry_info%mean_cell_area

        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if self.itime_scheme >= 4:
            self.ntl1 = nnow
            self.ntl2 = nnew
        else:
            self.ntl1 = 1
            self.ntl2 = 1

        self.run_predictor_step()

        self.run_corrector_step()

        if self.icon_grid.limited_area() or jg > 1:
            nhsolve_prog.stencils_66_67(
                self.metric_state.bdy_halo_c,  # TODO: @abishekg7 check if this should be mask_prog_halo_c_dsl_low_refin
                prognostic_state[nnew].rho,
                prognostic_state[nnew].theta_v,
                prognostic_state[nnew].exner,
                self.rd_o_cvd,
                self.rd_o_p0ref,
                cell_startindex_interior - 1,
                cell_endindex_local,
                cell_endindex_nudging,
                self.grid.n_lev(),
                offset_provider={},
            )
            # _mo_solve_nonhydro_stencil_66()
            # _mo_solve_nonhydro_stencil_67()

        mo_solve_nonhydro_stencil_68(
            self.metric_state.mask_prog_halo_c,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnew].exner,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].theta_v,
            constants.CVD_O_RD,
            cell_startindex_interior - 1,
            cell_endindex_local,
            self.grid.n_lev(),
            offset_provider={},
        )

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: list[PrognosticState],
        config: NonHydrostaticConfig,
        z_fields: ZFields,
        inv_dual_edge_length: Field[[EdgeDim], float],
        primal_normal_cell_1: Field[[ECDim], float],
        dual_normal_cell_1: Field[[ECDim], float],
        primal_normal_cell_2: Field[[ECDim], float],
        dual_normal_cell_2: Field[[ECDim], float],
        dtime: float,
        idyn_timestep: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
    ):
        if config.itime_scheme >= 6 or l_init or l_recompute:
            if config.itime_scheme < 6 and not l_init:
                lvn_only = True  # Recompute only vn tendency
            else:
                lvn_only = False
            velocity_advection.run_predictor_step()
        nvar = nnow

        p_dthalf = 0.5 * dtime

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
        ) = self.init_dimensions_boundaries()

        # initialize nest boundary points of z_rth_pr with zero
        if jg > 1 or self.icon_grid.limited_area():
            nhsolve_prog.mo_solve_nonhydro_stencil_01(
                self.z_rth_pr_1,
                self.z_rth_pr_2,
                cell_endindex_local,
                self.grid.n_lev(),
                offset_provider={}
            )
            # _mo_solve_nonhydro_stencil_01()

        nhsolve_prog.predictor_stencils_2_3(
            self.metric_state.exner_exfac,
            prognostic_state[nnow].exner,
            self.metric_state.exner_ref_mc,
            diagnostic_state.exner_pr,
            self.z_exner_ex_pr,
            cell_endindex_interior - 1,
            self.grid.n_lev(),
            offset_provider={},
        )
        # _mo_solve_nonhydro_stencil_02()
        # _mo_solve_nonhydro_stencil_03()

        if config.igradp_method <= 3:
            nhsolve_prog.predictor_stencils_4_5_6(
                self.metric_state.wgtfacq_c,
                self.z_exner_ex_pr,
                self.z_exner_ic,
                self.metric_state.wgtfac_c,
                self.metric_state.inv_ddqz_z_full,
                self.z_dexner_dz_c_1,
                cell_endindex_interior - 1,
                self.vertical_params.nflatlev,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )
            # _mo_solve_nonhydro_stencil_04()
            # _mo_solve_nonhydro_stencil_05()
            # _mo_solve_nonhydro_stencil_06()

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        nhsolve_prog.predictor_stencils_7_8_9(
            prognostic_state[nnow].rho,
            self.metric_state.rho_ref_mc,
            prognostic_state[nnow].theta_v,
            self.metric_state.theta_ref_mc,
            diagnostic_state.rho_ic,
            self.z_rth_pr_1,
            self.z_rth_pr_2,
            self.metric_state.wgtfac_c,
            self.metric_state.vwind_expl_wgt,
            diagnostic_state.exner_pr,
            self.metric_state.d_exner_dz_ref_ic,
            self.metric_state.ddqz_z_half,
            self.z_theta_v_pr_ic,
            diagnostic_state.theta_v_ic,
            self.z_th_ddz_exner_c,
            cell_endindex_interior - 1,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )
        # _mo_solve_nonhydro_stencil_07()
        # _mo_solve_nonhydro_stencil_08()
        # _mo_solve_nonhydro_stencil_09()

        if config.l_open_ubc and not l_vert_nested:
            raise NotImplementedError(
                "Nesting support not implemented. " "l_open_ubc not implemented"
            )

        # Perturbation theta at top and surface levels
        nhsolve_prog.predictor_stencils_11_lower_upper(
            self.metric_state.wgtfacq_c,
            self.z_rth_pr,
            self.metric_state.theta_ref_ic,
            self.z_theta_v_pr_ic,
            diagnostic_state.theta_v_ic,
            cell_endindex_interior - 1,
            self.grid.n_lev() + 1,
            offset_provider={"Koff": KDim},
        )
        # _mo_solve_nonhydro_stencil_11_lower()
        # _mo_solve_nonhydro_stencil_11_upper()

        if config.igradp_method <= 3:
            # Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
            nhsolve_prog.mo_solve_nonhydro_stencil_12(
                self.z_theta_v_pr_ic,
                self.metric_state.d2dexdz2_fac1_mc,
                self.metric_state.d2dexdz2_fac2_mc,
                self.z_rth_pr_2,
                self.z_dexner_dz_c_2,
                cell_endindex_interior - 1,
                self.grid.nflat_gradp(),
                self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme
        nhsolve_prog.mo_solve_nonhydro_stencil_13(
            prognostic_state[nnow].rho,
            self.metric_state.rho_ref_mc,
            prognostic_state[nnow].theta_v,
            self.metric_state.theta_ref_mc,
            self.z_rth_pr_1,
            self.z_rth_pr_2,
            cell_startindex_interior - 2,
            cell_endindex_interior - 2,
            self.grid.n_lev(),
            offset_provider={},
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if self.iadv_rhotheta == 1:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
            # mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
        elif self.iadv_rhotheta == 2:
            # Compute Green-Gauss gradients for rho and theta
            mo_math_gradients_grad_green_gauss_cell_dsl(
                p_grad_1_u,
                p_grad_1_v,
                p_grad_2_u,
                p_grad_2_v,
                p_ccpr1,
                p_ccpr2,
                geofac_grg_x,
                geofac_grg_y,
                cell_endindex_local,
                self.grid.n_lev(),
                offset_provider={
                    "C2E2CO": self.grid.get_c2e2co_connectivity(),
                    "C2E2CODim": C2E2CODim,
                },
            )
        elif self.iadv_rhotheta == 3:
            # First call: compute backward trajectory with wind at time level nnow
            lcompute = True
            lcleanup = False
            # upwind_hflux_miura3()
            # Second call: compute only reconstructed value for flux divergence
            lcompute = False
            lcleanup = True
            # upwind_hflux_miura3()

        # TODO: @abishekg7 Please see test.f90 for this section. Above call to 'wrap_run_mo_solve_nonhydro_stencil_14'
        if self.iadv_rhotheta <= 2:
            if config.idiv_method == 1:
                pass
            else:
                pass
        nhsolve_prog.mo_solve_nonhydro_stencil_14(
            self.z_rho_e,
            self.z_theta_v_e,
            edge_startindex_interior - 2,
            edge_endindex_interior - 3, #TODO: @abishekg7 conditional on idiv_method
            self.grid.n_lev(),
            offset_provider={},
        )
        # initialize also nest boundary points with zero
        if jg > 1 or self.icon_grid.limited_area():
            nhsolve_prog.mo_solve_nonhydro_stencil_15(
                self.z_rho_e,
                self.z_theta_v_e,
                edge_endindex_interior - 1,
                self.grid.n_lev(),
                offset_provider={},
            )

        if self.iadv_rhotheta == 2:
            # Operations from upwind_hflux_miura are inlined in order to process both fields in one step
            pass
        else:
            # Compute upwind-biased values for rho and theta starting from centered differences
            # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
            # at a second-order accurate FV discretization, but twice the length is needed for numerical stability
            nhsolve_prog.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
                prognostic_state[nnow].vn,
                diagnostic_state.vt,
                self.interpolation_state.pos_on_tplane_e_1,
                self.interpolation_state.pos_on_tplane_e_2,
                primal_normal_cell_1,
                dual_normal_cell_1,
                primal_normal_cell_2,
                dual_normal_cell_2,
                p_dthalf,
                self.metric_state.rho_ref_me,
                self.metric_state.theta_ref_me,
                self.z_grad_rth_1,
                self.z_grad_rth_2,
                self.z_grad_rth_3,
                self.z_grad_rth_4,
                self.z_rth_pr_1,
                self.z_rth_pr_2,
                edge_endindex_interior - 1,
                self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2CDim": E2CDim,
                },
            )

        # Remaining computations at edge points
        nhsolve_prog.mo_solve_nonhydro_stencil_18(
            inv_dual_edge_length,
            self.z_exner_ex_pr,
            self.z_gradh_exner,
            edge_startindex_nudging + 1,
            edge_endindex_interior,
            self.vertical_params.nflatlev - 1,
            offset_provider={"E2C": self.grid.get_e2c_connectivity(), "E2CDim": E2CDim},
        )

        if (
            config.igradp_method <= 3
        ):  # stencil_20 is tricky, there's no regular field operator
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction
            nhsolve_prog.nhsolve_predictor_tendencies_19_20(
                inv_dual_edge_length,
                self.z_exner_ex_pr,
                self.metric_state.ddxn_z_full,
                self.interpolation_state.c_lin_e,
                self.z_dexner_dz_c_1,
                self.z_gradh_exner,
                edge_startindex_nudging + 1,
                edge_endindex_interior,
                self.vertical_params.nflatlev,
                self.grid.nflat_gradp(),
                offset_provider={},
            )
            # mo_solve_nonhydro_stencil_19()
            # mo_solve_nonhydro_stencil_20()
        elif config.igradp_method == 4 or config.igradp_method == 5:
            # horizontal gradient of Exner pressure, cubic/quadratic interpolation
            raise NotImplementedError("igradp_method 4 and 5 not implemented")

        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if config.igradp_method == 3:
            mo_solve_nonhydro_stencil_21()
        elif config.igradp_method == 5:
            raise NotImplementedError("igradp_method 4 and 5 not implemented")

        if config.igradp_method == 3 or config.igradp_method == 5:
            nhsolve_prog.mo_solve_nonhydro_stencil_22(
                self.metric_state.ipeidx_dsl,
                self.metric_state.pg_exdist,
                self.z_hydro_corr,
                self.z_gradh_exner,
                edge_startindex_nudging + 1,
                edge_endindex_local,
                self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.mo_solve_nonhydro_stencil_24(
            prognostic_state[nnow].vn,
            ddt_vn_adv_ntl1,
            diagnostic_state.ddt_vn_phy,
            self.z_theta_v_e,
            self.z_gradh_exner,
            prognostic_state[nnew].vn,
            dtime,
            constants.CPD,
            edge_startindex_nudging + 1,
            edge_endindex_interior,
            self.grid.n_lev(),
            offset_provider={},
        )

        if config.is_iau_active:
            nhsolve_prog.mo_solve_nonhydro_stencil_28(
                diagnostic_state.vn_incr,
                prognostic_state[nnew].vn,
                config.iau_wgt_dyn,
                edge_startindex_nudging + 1,
                edge_endindex_local,
                self.grid.n_lev(),
                offset_provider={},
            )

        if self.icon_grid.limited_area() or jg > 1:
            nhsolve_prog.mo_solve_nonhydro_stencil_29(
                diagnostic_state.grf_tend_vn,
                prognostic_state[nnow].vn,
                prognostic_state[nnew].vn,
                dtime,
                edge_endindex_nudging,
                self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        # TODO: @abishekg7 fill this up
        nhsolve_prog.mo_solve_nonhydro_stencil_30(
            edge_endindex_interior - 2,
            self.grid.n_lev(),
        )

        #####  Not sure about  _mo_solve_nonhydro_stencil_31()

        if config.idiv_method == 1:
            nhsolve_prog.mo_solve_nonhydro_stencil_32(
                self.z_rho_e,
                self.z_vn_avg,
                self.metric_state.ddqz_z_full_e,
                self.z_theta_v_e,
                diagnostic_state.mass_fl_e,
                self.z_theta_v_fl_e,
                edge_endindex_interior - 2,
                self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.predictor_stencils_35_36(
            prognostic_state[nnew].vn,
            self.metric_state.ddxn_z_full,
            self.metric_state.ddxt_z_full,
            diagnostic_state.vt,
            z_fields.z_w_concorr_me,
            self.metric_state.wgtfac_e,
            diagnostic_state.vn_ie,
            z_fields.z_vt_ie,
            z_fields.z_kin_hor_e,
            edge_endindex_interior - 2,
            self.vertical_params.nflatlev,
            self.grid.n_lev(),
            offset_provider={},
        )
        # _mo_solve_nonhydro_stencil_35()
        # _mo_solve_nonhydro_stencil_36()

        if not l_vert_nested:
            nhsolve_prog.predictor_stencils_37_38(
                prognostic_state[nnew].vn,
                diagnostic_state.vt,
                diagnostic_state.vn_ie,
                z_fields.z_vt_ie,
                z_fields.z_kin_hor_e,
                self.metric_state.wgtfacq_e,
                edge_endindex_local - 2,
                self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )
            # _mo_solve_nonhydro_stencil_37()
            # _mo_solve_nonhydro_stencil_38()

        nhsolve_prog.predictor_stencils_39_40(
            self.interpolation_state.e_bln_c_s,
            z_fields.z_w_concorr_me,
            self.metric_state.wgtfac_c,
            self.metric_state.wgtfacq_c,
            diagnostic_state.w_concorr_c,
            cell_endindex_local - 1,
            self.vertical_params.nflatlev + 1,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2EDim": C2EDim,
                "Koff": KDim,
            },
        )
        # _mo_solve_nonhydro_stencil_39()
        # _mo_solve_nonhydro_stencil_40()

        if config.idiv_method == 2:
            if self.icon_grid.limited_area():
                init_zero_contiguous_dp()

            ##stencil not translated

        if config.idiv_method == 2:
            div_avg()

        if config.idiv_method == 1:
            nhsolve_prog.mo_solve_nonhydro_stencil_41(
                self.interpolation_state.geofac_div,
                diagnostic_state.mass_fl_e,
                self.z_theta_v_fl_e,
                self.z_flxdiv_mass,
                self.z_flxdiv_theta,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                },
            )

        # check for p_nh%prog(nnow)% fields and new
        nhsolve_prog.stencils_43_44_45_45b(
            self.z_w_expl,
            prognostic_state[nnow].w,
            diagnostic_state.ddt_w_adv_pc[ntl1],  # =ddt_w_adv_ntl1
            self.z_th_ddz_exner_c,
            self.z_contr_w_fl_l,
            diagnostic_state.rho_ic,
            diagnostic_state.w_concorr_c,
            self.metric_state.vwind_expl_wgt,
            self.z_beta,
            prognostic_state[nnow].exner,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            self.metric_state.inv_ddqz_z_full,
            self.z_alpha,
            self.metric_state.vwind_impl_wgt,
            diagnostic_state.theta_v_ic,
            self.z_q,
            constants.RD,
            constants.CVD,
            dtime,
            constants.CPD,
            cell_startindex_nudging + 1,
            cell_endindex_local,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={},
        )
        # _mo_solve_nonhydro_stencil_43()
        # _mo_solve_nonhydro_stencil_44()
        # _mo_solve_nonhydro_stencil_45()
        # _mo_solve_nonhydro_stencil_45_b()

        if not (config.l_open_ubc and l_vert_nested):
            nhsolve_prog.mo_solve_nonhydro_stencil_46(
                prognostic_state[nnew].w,
                self.z_contr_w_fl_l,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                offset_provider={},
            )

        nhsolve_prog.stencils_47_48_49(
            prognostic_state[nnew].w,
            self.z_contr_w_fl_l,
            diagnostic_state.w_concorr_c,
            self.z_rho_expl,
            self.z_exner_expl,
            prognostic_state[nnow].rho,
            self.metric_state.inv_ddqz_z_full,
            self.z_flxdiv_mass,
            diagnostic_state.exner_pr,
            self.z_beta,
            self.z_flxdiv_theta,
            diagnostic_state.theta_v_ic,
            diagnostic_state.ddt_exner_phy,
            dtime,
            offset_provider={},
        )
        # _mo_solve_nonhydro_stencil_47()
        # _mo_solve_nonhydro_stencil_48()
        # _mo_solve_nonhydro_stencil_49()

        if config.is_iau_active:
            nhsolve_prog.mo_solve_nonhydro_stencil_50(
                self.z_rho_expl,
                self.z_exner_expl,
                diagnostic_state.rho_incr,
                diagnostic_state.exner_incr,
                config.iau_wgt_dyn,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.stencils_52_53(
            self.metric_state.vwind_impl_wgt,
            diagnostic_state.theta_v_ic,
            self.metric_state.ddqz_z_half,
            self.z_alpha,
            self.z_beta,
            self.z_w_expl,
            self.z_exner_expl,
            self.z_q,
            prognostic_state[nnew].w,
            dtime,
            constants.CPD,
            cell_startindex_nudging + 1,
            cell_endindex_local,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )
        # _mo_solve_nonhydro_stencil_52()
        # _mo_solve_nonhydro_stencil_53()

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            nhsolve_prog.mo_solve_nonhydro_stencil_54(
                z_raylfac,
                prognostic_state[nnew].w_1,
                prognostic_state[nnew].w,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                offset_provider={},
            )

        nhsolve_prog.mo_solve_nonhydro_stencil_55(
            self.z_rho_expl,
            self.metric_state.vwind_impl_wgt,
            self.metric_state.inv_ddqz_z_full,
            diagnostic_state.rho_ic,
            prognostic_state[nnew].w,
            self.z_exner_expl,
            self.metric_state.exner_ref_mc,
            self.z_alpha,
            self.z_beta,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].exner,
            prognostic_state[nnew].theta_v,
            dtime,
            constants.CVD_O_RD,
            cell_startindex_nudging + 1,
            cell_endindex_local,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        # compute dw/dz for divergence damping term
        if config.lhdiff_rcf and config.divdamp_type >= 3:
            nhsolve_prog.mo_solve_nonhydro_stencil_56_63(
                self.metric_state.inv_ddqz_z_full,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.z_dwdz_dd,
                cell_startindex_nudging + 1,
                cell_endindex_local,
                config.kstart_dd3d,  # TODO: @abishekg7
                self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        if idyn_timestep == 1:
            nhsolve_prog.predictor_stencils_59_60(
                prognostic_state[nnow].exner,
                prognostic_state[nnew].exner,
                self.exner_dyn_incr,
                diagnostic_state.ddt_exner_phy,
                config.ndyn_substeps_var,
                dtime,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                config.kstart_moist,  # TODO: @abishekg7
                self.grid.n_lev(),
                offset_provider={},
            )
            # _mo_solve_nonhydro_stencil_59()
            # _mo_solve_nonhydro_stencil_60()

        if self.icon_grid.limited_area():  # for MPI-parallelized case
            nhsolve_prog.predictor_stencils_61_62(
                prognostic_state[nnow].rho,
                diagnostic_state.grf_tend_rho,
                prognostic_state[nnow].theta_v,
                diagnostic_state.grf_tend_thv,
                prognostic_state[nnow].w,
                diagnostic_state.grf_tend_w,
                prognostic_state[nnew].rho,
                prognostic_state[nnew].exner,
                prognostic_state[nnew].w,
                dtime,
                cell_endindex_nudging,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )
            # _mo_solve_nonhydro_stencil_61()
            # _mo_solve_nonhydro_stencil_62()

        if config.lhdiff_rcf and config.divdamp_type >= 3:
            nhsolve_prog.mo_solve_nonhydro_stencil_56_63(
                self.metric_state.inv_ddqz_z_full,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.z_dwdz_dd,
                cell_endindex_nudging, #TODO: @abishekg7 double check domains
                config.kstart_dd3d,  # TODO: @abishekg7
                self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        ##### COMMUNICATION PHASE

    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
        config: NonHydrostaticConfig,
        lclean_mflx,
        nnew,
    ):

        (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
        ) = self.init_dimensions_boundaries()

        lvn_only = False
        velocity_advection.run_corrector_step()
        nvar = nnew

        nhsolve_prog.mo_solve_nonhydro_stencil_10(
            prognostic_state.w,
            diagnostic_state.w_concorr_c,
            self.metric_state.ddqz_z_half,
            prognostic_state_now.rho,
            prognostic_state_var.rho,
            prognostic_state_now.theta_v,
            prognostic_state_var.theta_v,
            self.metric_state.wgtfac_c,
            self.metric_state.theta_ref_mc,
            self.metric_state.vwind_expl_wgt,
            diagnostic_state.exner_pr,
            self.metric_state.d_exner_dz_ref_ic,
            diagnostic_state.rho_ic,
            self.z_theta_v_pr_ic,
            diagnostic_state.theta_v_ic,
            self.z_th_ddz_exner_c,
            dtime,
            wgt_nnow_rth,
            wgt_nnew_rth,
            cell_endindex_interior,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.l_open_ubc and not l_vert_nested:
            raise NotImplementedError("l_open_ubc not implemented")

        nhsolve_prog.mo_solve_nonhydro_stencil_17(
            self.metric_state.hmask_dd3d,
            self.metric_state.scalfac_dd3d,
            inv_dual_edge_length,
            z_dwdz_dd,
            z_graddiv_vn,
            edge_endindex_interior - 2,
            config.kstart_dd3d,  # TODO: @abishekg7 resolve
            self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_offset_provider(),
            },
        )

        if config.itime_scheme >= 4:
            nhsolve_prog.mo_solve_nonhydro_stencil_23(
                prognostic_state_now.vn,
                diagnostic_state.ddt_vn_adv_ntl1,
                diagnostic_state.ddt_vn_adv_ntl2,
                diagnostic_state.ddt_vn_phy,
                self.z_theta_v_e,
                self.z_gradh_exner,
                prognostic_state_new.vn,
                dtime,
                wgt_nnow_vel,
                wgt_nnew_vel,
                cpd,
                edge_startindex_nudging + 1,
                edge_endindex_interior,
                self.grid.n_lev(),
                offset_provider={},
            )

        if lhdiff_rcf and (config.divdamp_order == 4 or config.divdamp_order == 24):
            nhsolve_prog.mo_solve_nonhydro_stencil_25(
                self.interpolation_state.geofac_grdiv,
                z_graddiv_vn,
                z_graddiv2_vn,
                edge_startindex_nudging + 1,
                edge_endindex_interior,
                self.grid.n_lev(),
                offset_provider={
                    "E2C2EO": self.grid.get_e2c2eO_offset_provider(),
                    "E2C2EODim": E2C2EODim,
                },
            )

        if lhdiff_rcf:
            if config.divdamp_order == 2 or (
                config.divdamp_order == 24 and scal_divdamp_o2 > 1e-6
            ):
                nhsolve_prog.mo_solve_nonhydro_stencil_26(
                    z_graddiv_vn,
                    prognostic_state_new.vn,
                    scal_divdamp_o2,
                    edge_startindex_nudging + 1,
                    edge_endindex_local,
                    self.grid.n_lev(),
                    offset_provider={},
                )
            if config.divdamp_order == 4 or (
                config.divdamp_order == 24 and ivdamp_fac_o2 <= 4 * divdamp_fac
            ):
                if self.icon_grid.limited_area():
                    nhsolve_prog.mo_solve_nonhydro_stencil_27(
                        scal_divdamp,
                        bdy_divdamp,
                        self.interpolation_state.nudgecoeff_e,
                        z_graddiv2_vn,
                        prognostic_state_new.vn,
                        edge_startindex_nudging + 1,
                        edge_endindex_interior,
                        self.grid.n_lev(),
                        offset_provider={},
                    )
                else:
                    nhsolve_prog.mo_solve_nonhydro_4th_order_divdamp(
                        scal_divdamp,
                        z_graddiv2_vn,
                        prognostic_state_new.vn,
                        edge_startindex_nudging + 1,
                        edge_endindex_interior,
                        self.grid.n_lev(),
                        offset_provider={},
                    )

        if config.is_iau_active:
            nhsolve_prog.mo_solve_nonhydro_stencil_28(
                diagnostic_state.vn_incr,
                prognostic_state_new.vn,
                iau_wgt_dyn,
                edge_startindex_nudging + 1,
                edge_endindex_interior,
                self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        if config.idiv_method == 1:
            nhsolve_prog.mo_solve_nonhydro_stencil_32(
                self.z_rho_e,
                self.z_vn_avg,
                self.metric_state.ddqz_z_full_e,
                self.z_theta_v_e,
                diagnostic_state.mass_fl_e,
                self.z_theta_v_fl_e,
                edge_endindex_local - 2,
                self.grid.n_lev(),
                offset_provider={},
            )

            # TODO: @abishekg7 resolve prep_adv
            if lpred_adv:  # Preparations for tracer advection
                if lclean_mflx:
                    nhsolve_prog.mo_solve_nonhydro_stencil_33(
                        prep_adv.vn_traj,
                        prep_adv.mass_flx_me,
                        edge_endindex_local,
                        self.grid.n_lev(),
                        offset_provider={},
                    )
                nhsolve_prog.mo_solve_nonhydro_stencil_34(
                    z_vn_avg,
                    diagnostic_state.mass_fl_e,
                    prep_adv.vn_traj,
                    prep_adv.mass_flx_me,
                    r_nsubsteps,
                    edge_endindex_interior - 2,
                    self.grid.n_lev(),
                    offset_provider={},
                )

        if config.itime_scheme >= 5:
            nhsolve_prog.corrector_stencils_35_39_40( #TODO: @abishekg7 bounds are complicated
                prognostic_state[nnew].vn,
                self.metric_state.ddxn_z_full,
                self.metric_state.ddxt_z_full,
                diagnostic_state.vt,
                z_fields.z_w_concorr_me,
                self.interpolation_state.e_bln_c_s,
                self.metric_state.wgtfac_c,
                self.metric_state.wgtfacq_c,
                diagnostic_state.w_concorr_c,
                edge_endindex_local - 2,
                cell_endindex_local - 1,
                self.vertical_params.nflatlev + 1,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                self.vertical_params.nflatlev,
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                    "Koff": KDim,
                },
            )
            # _mo_solve_nonhydro_stencil_35()
            # _mo_solve_nonhydro_stencil_39()
            # _mo_solve_nonhydro_stencil_40()

        if config.idiv_method == 2:
            if self.icon_grid.limited_area():
                init_zero_contiguous_dp()

            ##stencil not translated

        if config.idiv_method == 2:
            div_avg()

        if config.idiv_method == 1:
            nhsolve_prog.mo_solve_nonhydro_stencil_41(
                self.interpolation_state.geofac_div,
                diagnostic_state.mass_fl_e,
                self.z_theta_v_fl_e,
                self.z_flxdiv_mass,
                self.z_flxdiv_theta,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2EDim": C2EDim,
                },
            )

        # TODO: @abishekg7 45_b is just a zero stencil
        if config.itime_scheme >= 4:
            nhsolve_prog.stencils_42_44_45_45b(
                self.z_w_expl,
                prognostic_state[nnow].w,
                diagnostic_state.ddt_w_adv_pc[ntl1],  # =ddt_w_adv_ntl1
                diagnostic_state.ddt_w_adv_pc[ntl2],  # =ddt_w_adv_ntl2
                self.z_th_ddz_exner_c,
                self.z_contr_w_fl_l,
                diagnostic_state.rho_ic,
                diagnostic_state.w_concorr_c,
                self.metric_state.vwind_expl_wgt,
                self.z_beta,
                prognostic_state[nnow].exner,
                prognostic_state[nnow].rho,
                prognostic_state[nnow].theta_v,
                self.metric_state.inv_ddqz_z_full,
                self.z_alpha,
                self.metric_state.vwind_impl_wgt,
                diagnostic_state.theta_v_ic,
                self.z_q,
                constants.RD,
                constants.CVD,
                dtime,
                constants.CPD,
                wgt_nnow_vel,
                wgt_nnew_vel,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )
        else:
            nhsolve_prog.stencils_43_44_45_45b(
                self.z_w_expl,
                prognostic_state[nnow].w,
                diagnostic_state.ddt_w_adv_pc[ntl1],  # =ddt_w_adv_ntl1
                self.z_th_ddz_exner_c,
                self.z_contr_w_fl_l,
                diagnostic_state.rho_ic,
                diagnostic_state.w_concorr_c,
                self.metric_state.vwind_expl_wgt,
                self.z_beta,
                prognostic_state[nnow].exner,
                prognostic_state[nnow].rho,
                prognostic_state[nnow].theta_v,
                self.metric_state.inv_ddqz_z_full,
                self.z_alpha,
                self.metric_state.vwind_impl_wgt,
                diagnostic_state.theta_v_ic,
                self.z_q,
                constants.RD,
                constants.CVD,
                dtime,
                constants.CPD,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                self.grid.n_lev(),
                self.grid.n_lev() + 1,
                offset_provider={},
            )
            # mo_solve_nonhydro_stencil_43()

        # mo_solve_nonhydro_stencil_44()
        # mo_solve_nonhydro_stencil_45()
        # mo_solve_nonhydro_stencil_45_b()

        if not config.l_open_ubc and not l_vert_nested:
            nhsolve_prog.mo_solve_nonhydro_stencil_46(
                prognostic_state[nnew].w,
                self.z_contr_w_fl_l,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                offset_provider={},
            )

        nhsolve_prog.stencils_47_48_49(
            prognostic_state[nnew].w,
            self.z_contr_w_fl_l,
            diagnostic_state.w_concorr_c,
            self.z_rho_expl,
            self.z_exner_expl,
            prognostic_state[nnow].rho,
            self.metric_state.inv_ddqz_z_full,
            self.z_flxdiv_mass,
            diagnostic_state.exner_pr,
            self.z_beta,
            self.z_flxdiv_theta,
            diagnostic_state.theta_v_ic,
            diagnostic_state.ddt_exner_phy,
            dtime,
            cell_startindex_nudging + 1,
            cell_endindex_interior,
            self.grid.n_lev(),
            self.grid.n_lev() + 1,
            offset_provider={"Koff": KDim},
        )
        # _mo_solve_nonhydro_stencil_47()
        # _mo_solve_nonhydro_stencil_48()
        # _mo_solve_nonhydro_stencil_49()

        if config.is_iau_active:
            nhsolve_prog.mo_solve_nonhydro_stencil_50(
                self.z_rho_expl,
                self.z_exner_expl,
                diagnostic_state.rho_incr,
                diagnostic_state.exner_incr,
                config.iau_wgt_dyn,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.stencils_52_53(
            self.metric_state.vwind_impl_wgt,
            diagnostic_state.theta_v_ic,
            self.metric_state.ddqz_z_half,
            self.z_alpha,
            self.z_beta,
            self.z_w_expl,
            self.z_exner_expl,
            self.z_q,
            prognostic_state[nnew].w,
            dtime,
            constants.CPD,
            cell_startindex_nudging,
            cell_endindex_interior,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )
        # _mo_solve_nonhydro_stencil_52()
        # _mo_solve_nonhydro_stencil_53()

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            nhsolve_prog.mo_solve_nonhydro_stencil_54(
                z_raylfac,
                prognostic_state[nnew].w_1,
                prognostic_state[nnew].w,
                cell_startindex_nudging + 1,
                cell_endindex_interior,
                offset_provider={},
            )

        nhsolve_prog.mo_solve_nonhydro_stencil_55(
            self.z_rho_expl,
            self.metric_state.vwind_impl_wgt,
            self.metric_state.inv_ddqz_z_full,
            diagnostic_state.rho_ic,
            prognostic_state[nnew].w,
            self.z_exner_expl,
            self.metric_state.exner_ref_mc,
            self.z_alpha,
            self.z_beta,
            prognostic_state[nnow].rho,
            prognostic_state[nnow].theta_v,
            prognostic_state[nnow].exner,
            prognostic_state[nnew].rho,
            prognostic_state[nnew].exner,
            prognostic_state[nnew].theta_v,
            dtime,
            constants.CVD_O_RD,
            cell_startindex_nudging + 1,
            cell_endindex_interior,
            self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if lpred_adv:
            if lclean_mflx:
                # TODO: @abishekg7 domains for this
                set_zero_c_k(self.mass_flx_ic,
                             cell_startindex_nudging + 1,
                             cell_endindex_interior,
                             self.grid.n_lev(),
                             offset_provider={})
                # mo_solve_nonhydro_stencil_57()

        nhsolve_prog.mo_solve_nonhydro_stencil_58(
            self.z_contr_w_fl_l,
            diagnostic_state.rho_ic,
            self.metric_state.vwind_impl_wgt,
            prognostic_state[nnew].w,
            mass_flx_ic,
            r_nsubsteps,
            cell_startindex_nudging + 1,
            cell_endindex_interior,
            self.grid.n_lev(),
            offset_provider={},
        )

        if lpred_adv:
            if lclean_mflx:
                # TODO: @abishekg7 domains for this
                set_zero_c_k(self.mass_flx_ic,
                             cell_endindex_nudging,
                             self.grid.n_lev()+1,
                             offset_provider={})
                # mo_solve_nonhydro_stencil_64()

            nhsolve_prog.mo_solve_nonhydro_stencil_65(
                diagnostic_state.rho_ic,
                self.metric_state.vwind_expl_wgt,
                self.metric_state.vwind_impl_wgt,
                prognostic_state[nnow].w,
                prognostic_state[nnew].w,
                diagnostic_state.w_concorr_c,
                self.mass_flx_ic,
                r_nsubsteps,
                cell_endindex_nudging,
                self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE



    def init_dimensions_boundaries(self):
        (
            edge_startindex_nudging,
            edge_endindex_nudging,
        ) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim),
            HorizontalMarkerIndex.nudging(EdgeDim),
        )

        (edge_startindex_interior, edge_endindex_interior,) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.interior(EdgeDim),
            HorizontalMarkerIndex.interior(EdgeDim),
        )

        (edge_startindex_local, edge_endindex_local,) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.local(EdgeDim),
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (
            cell_startindex_nudging,
            cell_endindex_nudging,
        ) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.nudging(CellDim),
        )

        (cell_startindex_interior, cell_endindex_interior,) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.interior(CellDim),
            HorizontalMarkerIndex.interior(CellDim),
        )

        (cell_startindex_local, cell_endindex_local,) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        return (
            edge_startindex_nudging,
            edge_endindex_nudging,
            edge_startindex_interior,
            edge_endindex_interior,
            edge_startindex_local,
            edge_endindex_local,
            cell_startindex_nudging,
            cell_endindex_nudging,
            cell_startindex_interior,
            cell_endindex_interior,
            cell_startindex_local,
            cell_endindex_local,
        )
