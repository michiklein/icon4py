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
from typing import Final, Optional

import numpy as np
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn

import icon4py.common.constants as constants
import icon4py.nh_solve.solve_nonhydro_program as nhsolve_prog
from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.atm_dyn_iconam.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_01 import (
    mo_solve_nonhydro_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_10 import (
    mo_solve_nonhydro_stencil_10,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_12 import (
    mo_solve_nonhydro_stencil_12,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_13 import (
    mo_solve_nonhydro_stencil_13,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_17 import (
    mo_solve_nonhydro_stencil_17,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_18 import (
    mo_solve_nonhydro_stencil_18,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_19 import (
    mo_solve_nonhydro_stencil_19,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_20 import (
    mo_solve_nonhydro_stencil_20,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_21 import (
    mo_solve_nonhydro_stencil_21,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_22 import (
    mo_solve_nonhydro_stencil_22,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_23 import (
    mo_solve_nonhydro_stencil_23,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_24 import (
    mo_solve_nonhydro_stencil_24,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_25 import (
    mo_solve_nonhydro_stencil_25,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_26 import (
    mo_solve_nonhydro_stencil_26,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_27 import (
    mo_solve_nonhydro_stencil_27,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_28 import (
    mo_solve_nonhydro_stencil_28,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_29 import (
    mo_solve_nonhydro_stencil_29,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_30 import (
    mo_solve_nonhydro_stencil_30,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_31 import (
    mo_solve_nonhydro_stencil_31,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_32 import (
    mo_solve_nonhydro_stencil_32,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_33 import (
    mo_solve_nonhydro_stencil_33,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_34 import (
    mo_solve_nonhydro_stencil_34,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_41 import (
    mo_solve_nonhydro_stencil_41,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_46 import (
    mo_solve_nonhydro_stencil_46,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_50 import (
    mo_solve_nonhydro_stencil_50,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_52 import (
    mo_solve_nonhydro_stencil_52,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_53 import (
    mo_solve_nonhydro_stencil_53,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_54 import (
    mo_solve_nonhydro_stencil_54,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_55 import (
    mo_solve_nonhydro_stencil_55,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_56_63 import (
    mo_solve_nonhydro_stencil_56_63,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_58 import (
    mo_solve_nonhydro_stencil_58,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_59 import (
    mo_solve_nonhydro_stencil_59,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_65 import (
    mo_solve_nonhydro_stencil_65,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_66 import (
    mo_solve_nonhydro_stencil_66,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_67 import (
    mo_solve_nonhydro_stencil_67,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_68 import (
    mo_solve_nonhydro_stencil_68,
)
from icon4py.common.dimension import CellDim, ECDim, EdgeDim, KDim, VertexDim
from icon4py.grid.horizontal import EdgeParams, HorizontalMarkerIndex
from icon4py.grid.icon_grid import IconGrid
from icon4py.grid.vertical import VerticalModelParams
from icon4py.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.state_utils.interpolation_state import InterpolationState
from icon4py.state_utils.metric_state import MetricStateNonHydro
from icon4py.state_utils.nh_constants import NHConstants
from icon4py.state_utils.prep_adv_state import PrepAdvection
from icon4py.state_utils.prognostic_state import PrognosticState
from icon4py.state_utils.utils import (
    _allocate,
    _allocate_indices,
    _calculate_bdy_divdamp,
    _en_smag_fac_for_zero_nshift,
    compute_z_raylfac,
    scal_divdamp_calcs,
    set_zero_c_k,
    set_zero_e_k,
)
from icon4py.state_utils.z_fields import ZFields
from icon4py.velocity import velocity_advection


class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    def __init__(
        self,
        itime_scheme: int = 4,
        iadv_rhotheta: int = 2,
        igradp_method: int = 3,
        ndyn_substeps_var: float = 2.0,
        rayleigh_type: int = 2,
        divdamp_order: int = 24,
        idiv_method: int = 1,
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 1.0,
        divdamp_type: int = 3,
        lhdiff_rcf: bool = True,
        l_vert_nested: bool = False,
        l_open_ubc: bool = False,
        rhotheta_offctr: float = -0.1,
        veladv_offctr: float = 0.25,
        divdamp_fac: float = 0.004,  # checked for corrector after serialization
        divdamp_fac_o2: float = 0.032,  # checked for corrector after serialization
        max_nudging_coeff: float = 0.075,
        divdamp_fac2: float = 0.004,
        divdamp_fac3: float = 0.004,
        divdamp_fac4: float = 0.004,
        divdamp_z: float = 32500,
        divdamp_z2: float = 40000,
        divdamp_z3: float = 60000,
        divdamp_z4: float = 80000,
    ):

        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme
        self.iadv_rhotheta: int = iadv_rhotheta

        self.l_open_ubc: bool = l_open_ubc
        self.igradp_method: int = igradp_method
        self.ndyn_substeps_var = ndyn_substeps_var
        self.idiv_method: int = idiv_method
        self.is_iau_active: bool = is_iau_active
        self.iau_wgt_dyn: float = iau_wgt_dyn
        self.divdamp_type: int = divdamp_type
        self.lhdiff_rcf: bool = lhdiff_rcf
        self.rayleigh_type: int = rayleigh_type
        self.divdamp_order: int = divdamp_order
        self.l_vert_nested: bool = l_vert_nested
        self.rhotheta_offctr: float = rhotheta_offctr
        self.veladv_offctr: float = veladv_offctr
        self.divdamp_fac: float = divdamp_fac
        self.divdamp_fac_o2: float = divdamp_fac_o2
        self.nudge_max_coeff: float = max_nudging_coeff

        self.divdamp_fac2: float = divdamp_fac2
        self.divdamp_fac3: float = divdamp_fac3
        self.divdamp_fac4: float = divdamp_fac4
        self.divdamp_z: float = divdamp_z
        self.divdamp_z2: float = divdamp_z2
        self.divdamp_z3: float = divdamp_z3
        self.divdamp_z4: float = divdamp_z4

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""
        if self.l_open_ubc:
            raise NotImplementedError(
                "Open upper boundary conditions not supported"
                "(to allow vertical motions related to diabatic heating to extend beyond the model top)"
            )

        if self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method != 3:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != 4:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.idiv_method != 1:
            raise NotImplementedError("idiv_method can only be 1")

        if self.divdamp_order != 24:
            raise NotImplementedError("divdamp_order can only be 24")

        if self.divdamp_type == 32:
            raise NotImplementedError("divdamp_type with value 32 not yet implemented")


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):
        self.rd_o_cvd: Final[float] = constants.RD / constants.CVD
        self.cvd_o_rd: Final[float] = constants.CVD / constants.RD
        self.rd_o_p0ref: Final[float] = constants.RD / constants.P0REF
        self.grav_o_cpd: Final[float] = constants.GRAV / constants.CPD

        # start level for 3D divergence damping terms
        self.kstart_dd3d: int = 0

        # start level for moist physics processes (specified by htop_moist_proc)
        self.kstart_moist: int = 1
        if self.kstart_moist != 1:
            raise NotImplementedError("kstart_moist can only be 1")

        self.alin = (config.divdamp_fac2 - config.divdamp_fac) / (
            config.divdamp_z2 - config.divdamp_z
        )

        self.df32 = config.divdamp_fac3 - config.divdamp_fac2
        self.dz32 = config.divdamp_z3 - config.divdamp_z2
        self.df42 = config.divdamp_fac4 - config.divdamp_fac2
        self.dz42 = config.divdamp_z4 - config.divdamp_z2

        self.bqdr = (self.df42 * self.dz32 - self.df32 * self.dz42) / (
            self.dz32 * self.dz42 * (self.dz42 - self.dz32)
        )
        self.aqdr = self.df32 / self.dz32 - self.bqdr * self.dz32


class SolveNonhydro:
    def __init__(self):
        self._initialized = False
        self.grid: Optional[IconGrid] = None
        self.interpolation_state = None
        self.metric_state_nonhydro = None
        self.vertical_params: Optional[VerticalModelParams] = None

        self.config: Optional[NonHydrostaticConfig] = None
        self.params: Optional[NonHydrostaticParams] = None

        self.l_vert_nested: bool = False
        self.enh_divdamp_fac = None
        self.scal_divdamp: Field[[KDim], float] = None
        self.p_test_run = None
        self.ntl1 = 0
        self.ntl2 = 0

    def init(
        self,
        grid: IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        a_vec: Field[[KDim], float],
        enh_smag_fac: Field[[KDim], float],
        cell_areas: Field[[CellDim], float],
        fac: tuple,
        z: tuple,
    ):
        """
        Initialize NonHydrostatic granule with configuration.

        calculates all local fields that are used in nh_solve within the time loop
        """
        self.config: NonHydrostaticConfig = config
        self.params: NonHydrostaticParams = params
        self.grid = grid
        self.vertical_params = vertical_params
        self.metric_state_nonhydro: MetricStateNonHydro = metric_state_nonhydro
        self.interpolation_state: InterpolationState = interpolation_state

        self._allocate_local_fields()
        self._initialized = True

        if self.grid.lvert_nest():
            self.l_vert_nested = True
            self.jk_start = 1
        else:
            self.jk_start = 0

        out = enh_smag_fac
        _en_smag_fac_for_zero_nshift(
            a_vec, *fac, *z, out=enh_smag_fac, offset_provider={"Koff": KDim}
        )
        self.enh_divdamp_fac = enh_smag_fac

        cell_areas_avg = np.sum(cell_areas) / float(self.grid.num_cells())
        # TODO @tehrengruber: fix power
        scal_divdamp_calcs.with_backend(run_gtfn)(
            enh_smag_fac,
            out,
            cell_areas_avg,
            offset_provider={},
        )
        self.scal_divdamp = out
        self.p_test_run = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_exner_ex_pr = _allocate(CellDim, KDim, is_halfdim=True, mesh=self.grid)
        self.z_exner_ic = _allocate(CellDim, KDim, is_halfdim=True, mesh=self.grid)
        self.z_dexner_dz_c_1 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_theta_v_pr_ic = _allocate(CellDim, KDim, is_halfdim=True, mesh=self.grid)
        self.z_th_ddz_exner_c = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rth_pr_1 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rth_pr_2 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_1 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_2 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_3 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_grad_rth_4 = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_dexner_dz_c_2 = _allocate(CellDim, KDim, mesh=self.grid)
        self.exner_dyn_incr = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_hydro_corr = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_vn_avg = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_theta_v_fl_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_flxdiv_mass = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_flxdiv_theta = _allocate(CellDim, KDim, mesh=self.grid)
        self.z_rho_v = _allocate(VertexDim, KDim, mesh=self.grid)
        self.z_theta_v_v = _allocate(VertexDim, KDim, mesh=self.grid)
        self.z_graddiv2_vn = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.k_field = _allocate_indices(KDim, mesh=self.grid, is_halfdim=True)
        self.z_w_concorr_me = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_kin_hor_e = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_vt_ie = _allocate(EdgeDim, KDim, mesh=self.grid)
        self.z_hydro_corr_horizontal = _allocate(EdgeDim, mesh=self.grid)
        self.z_raylfac = _allocate(KDim, mesh=self.grid)

    def set_timelevels(self, nnow, nnew):
        #  Set time levels of ddt_adv fields for call to velocity_tendencies
        if self.config.itime_scheme == 4:
            self.ntl1 = nnow
            self.ntl2 = nnew
        else:
            self.ntl1 = 0
            self.ntl2 = 0

    def time_step(
        self,
        diagnostic_state_nh: DiagnosticStateNonHydro,
        prognostic_state_ls: list[PrognosticState],
        prep_adv: PrepAdvection,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        nh_constants: NHConstants,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        c_owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        bdy_divdamp: Field[[KDim], float],
        dtime: float,
        idyn_timestep: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
        lclean_mflx: bool,
        lprep_adv: bool,
    ):

        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / config.ndyn_substeps_var

        # Coefficient for reduced fourth-order divergence damping along nest boundaries
        _calculate_bdy_divdamp(
            self.scal_divdamp,
            config.nudge_max_coeff,
            constants.dbl_eps,
            out=bdy_divdamp,
            offset_provider={},
        )

        # scaling factor for second-order divergence damping: divdamp_fac_o2*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        cell_areas_avg = np.sum(cell_areas) / float(self.grid.num_cells())
        scal_divdamp_o2 = config.divdamp_fac_o2 * (cell_areas_avg)

        (indices_cells_1, indices_cells_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            HorizontalMarkerIndex.end(CellDim),
        )

        (indices_edges_1, indices_edges_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim),
            HorizontalMarkerIndex.local(EdgeDim),
        )

        if self.p_test_run:
            nhsolve_prog.init_test_fields.with_backend(run_gtfn)(
                z_fields.z_rho_e,
                z_fields.z_theta_v_e,
                z_fields.z_dwdz_dd,
                z_fields.z_graddiv_vn,
                indices_edges_1,
                indices_edges_2,
                indices_cells_1,
                indices_cells_2,
                self.grid.n_lev(),
                offset_provider={},
            )

        self.set_timelevels(nnow, nnew)

        # self.run_predictor_step(
        #     diagnostic_state_nh=diagnostic_state_nh,
        #     prognostic_state=prognostic_state_ls,
        #     config=config,
        #     params=params,
        #     edge_geometry=edge_geometry,
        #     z_fields=z_fields,
        #     cfl_w_limit=cfl_w_limit,
        #     scalfac_exdiff=scalfac_exdiff,
        #     cell_areas=cell_areas,
        #     owner_mask=c_owner_mask,
        #     f_e=f_e,
        #     area_edge=area_edge,
        #     dtime=dtime,
        #     idyn_timestep=idyn_timestep,
        #     l_recompute=l_recompute,
        #     l_init=l_init,
        #     nnow=nnow,
        #     nnew=nnew,
        # )

        # self.run_corrector_step(
        #     diagnostic_state_nh=diagnostic_state_nh,
        #     prognostic_state=prognostic_state_ls,
        #     config=config,
        #     params=params,
        #     edge_geometry=edge_geometry,
        #     z_fields=z_fields,
        #     prep_adv=prep_adv,
        #     dtime=dtime,
        #     nnew=nnew,
        #     nnow=nnow,
        #     cfl_w_limit=cfl_w_limit,
        #     scalfac_exdiff=scalfac_exdiff,
        #     cell_areas=cell_areas,
        #     owner_mask=c_owner_mask,
        #     f_e=f_e,
        #     area_edge=area_edge,
        #     lclean_mflx=lclean_mflx,
        #     nh_constants=nh_constants,
        #     bdy_divdamp=bdy_divdamp,
        #     lprep_adv=lprep_adv,
        # )

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim) - 1,
            HorizontalMarkerIndex.local(CellDim),
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            HorizontalMarkerIndex.nudging(CellDim) - 1,
        )

        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_66.with_backend(run_gtfn)(
                bdy_halo_c=self.metric_state_nonhydro.bdy_halo_c,
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=params.rd_o_cvd,
                rd_o_p0ref=params.rd_o_p0ref,
                horizontal_start=indices_0_1,
                horizontal_end=indices_0_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )



            mo_solve_nonhydro_stencil_67.with_backend(run_gtfn)(
                rho=prognostic_state_ls[nnew].rho,
                theta_v=prognostic_state_ls[nnew].theta_v,
                exner=prognostic_state_ls[nnew].exner,
                rd_o_cvd=params.rd_o_cvd,
                rd_o_p0ref=params.rd_o_p0ref,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )


        mo_solve_nonhydro_stencil_68.with_backend(run_gtfn)(
            mask_prog_halo_c=self.metric_state_nonhydro.mask_prog_halo_c,
            rho_now=prognostic_state_ls[nnow].rho,
            theta_v_now=prognostic_state_ls[nnow].theta_v,
            exner_new=prognostic_state_ls[nnew].exner,
            exner_now=prognostic_state_ls[nnow].exner,
            rho_new=prognostic_state_ls[nnew].rho,
            theta_v_new=prognostic_state_ls[nnew].theta_v,
            cvd_o_rd=params.cvd_o_rd,
            horizontal_start=indices_0_1,
            horizontal_end=indices_0_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

    def run_predictor_step(
        self,
        diagnostic_state_nh: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        dtime: float,
        idyn_timestep: float,
        l_recompute: bool,
        l_init: bool,
        nnow: int,
        nnew: int,
    ):
        if l_init or l_recompute:
            if config.itime_scheme == 4 and not l_init:
                lvn_only = True  # Recompute only vn tendency
            else:
                lvn_only = False


        velocity_advection.VelocityAdvection(
            self.grid,
            self.metric_state_nonhydro,
            self.interpolation_state,
            self.vertical_params,
        ).run_predictor_step(
            vn_only=lvn_only,
            diagnostic_state=diagnostic_state_nh,
            prognostic_state=prognostic_state[nnow],
            z_w_concorr_me=self.z_w_concorr_me,
            z_kin_hor_e=self.z_kin_hor_e,
            z_vt_ie=self.z_vt_ie,
            inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
            inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
            dtime=dtime,
            ntnd=self.ntl1,
            tangent_orientation=edge_geometry.tangent_orientation,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            cell_areas=cell_areas,
            owner_mask=owner_mask,
            f_e=f_e,
            area_edge=area_edge,
        )

        p_dthalf = 0.5 * dtime

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            HorizontalMarkerIndex.end(CellDim),
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        (indices_2_1, indices_2_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.local(CellDim) - 2,
            HorizontalMarkerIndex.local(CellDim) - 2,
        )

        (indices_3_1, indices_3_2) = self.grid.get_indices_from_to(
            VertexDim,
            HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,  # TODO: check
            HorizontalMarkerIndex.local(VertexDim) - 1,
        )

        (indices_4_1, indices_4_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim),
            HorizontalMarkerIndex.local(EdgeDim) - 1,
        )

        (indices_11_1, indices_11_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            HorizontalMarkerIndex.nudging(CellDim) - 1,
        )

        (indices_12_1, indices_12_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
            HorizontalMarkerIndex.local(EdgeDim) - 1,
        )

        (indices_5_1, indices_5_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (indices_5b_1, indices_5b_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.end(EdgeDim),
        )

        (indices_6_1, indices_6_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim),
            HorizontalMarkerIndex.nudging(EdgeDim),
        )

        (indices_7_1, indices_7_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (indices_9_1, indices_9_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
            HorizontalMarkerIndex.local(CellDim) - 1,
        )

        (indices_10_1, indices_10_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        #  Precompute Rayleigh damping factor
        compute_z_raylfac.with_backend(run_gtfn)(
            rayleigh_w=self.metric_state_nonhydro.rayleigh_w,
            dtime=dtime,
            z_raylfac=self.z_raylfac,
            offset_provider={},
        )

        # initialize nest boundary points of z_rth_pr with zero
        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_01.with_backend(run_gtfn)(
                z_rth_pr_1=self.z_rth_pr_1,
                z_rth_pr_2=self.z_rth_pr_2,
                horizontal_start=indices_0_1,
                horizontal_end=indices_0_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        nhsolve_prog.predictor_stencils_2_3.with_backend(run_gtfn)(
            exner_exfac=self.metric_state_nonhydro.exner_exfac,
            exner=prognostic_state[nnow].exner,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_exner_ex_pr=self.z_exner_ex_pr,
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            k_field=self.k_field,
            nlev=self.grid.n_lev(),
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={},
        )

        if config.igradp_method == 3:
            nhsolve_prog.predictor_stencils_4_5_6.with_backend(run_gtfn)(
                wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c_dsl,
                z_exner_ex_pr=self.z_exner_ex_pr,
                z_exner_ic=self.z_exner_ic,
                wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                k_field=self.k_field,
                nlev=self.grid.n_lev(),
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=max(1, self.vertical_params.nflatlev),
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )

            if self.vertical_params.nflatlev == 1:
                # Perturbation Exner pressure on top half level
                raise NotImplementedError("nflatlev=1 not implemented")

        nhsolve_prog.predictor_stencils_7_8_9.with_backend(run_gtfn)(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            k_field=self.k_field,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        # Perturbation theta at top and surface levels
        nhsolve_prog.predictor_stencils_11_lower_upper.with_backend(run_gtfn)(
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c_dsl,
            z_rth_pr=self.z_rth_pr_2,
            theta_ref_ic=self.metric_state_nonhydro.theta_ref_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            k_field=self.k_field,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_1_1,
            horizontal_end=indices_1_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={"Koff": KDim},
        )

        if config.igradp_method == 3:
            # Second vertical derivative of perturbation Exner pressure (hydrostatic approximation)
            mo_solve_nonhydro_stencil_12.with_backend(run_gtfn)(
                z_theta_v_pr_ic=self.z_theta_v_pr_ic,
                d2dexdz2_fac1_mc=self.metric_state_nonhydro.d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc=self.metric_state_nonhydro.d2dexdz2_fac2_mc,
                z_rth_pr_2=self.z_rth_pr_2,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=self.vertical_params.nflat_gradp,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        # Add computation of z_grad_rth (perturbation density and virtual potential temperature at main levels)
        # at outer halo points: needed for correct calculation of the upwind gradients for Miura scheme

        mo_solve_nonhydro_stencil_13.with_backend(run_gtfn)(
            rho=prognostic_state[nnow].rho,
            rho_ref_mc=self.metric_state_nonhydro.rho_ref_mc,
            theta_v=prognostic_state[nnow].theta_v,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            z_rth_pr_1=self.z_rth_pr_1,
            z_rth_pr_2=self.z_rth_pr_2,
            horizontal_start=indices_2_1,
            horizontal_end=indices_2_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if config.iadv_rhotheta == 1:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                run_gtfn
            )(
                p_cell_in=prognostic_state[nnow].rho,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_rho_v,
                horizontal_start=indices_3_1,
                horizontal_end=indices_3_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_cell_in,2)
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                run_gtfn
            )(
                p_cell_in=prognostic_state[nnow].theta_v,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_theta_v_v,
                horizontal_start=indices_3_1,
                horizontal_end=indices_3_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_cell_in,2)
                offset_provider={
                    "V2C": self.grid.get_v2c_connectivity(),
                },
            )
        elif config.iadv_rhotheta == 2:
            # Compute Green-Gauss gradients for rho and theta
            mo_math_gradients_grad_green_gauss_cell_dsl.with_backend(run_gtfn)(
                p_grad_1_u=self.z_grad_rth_1,
                p_grad_1_v=self.z_grad_rth_2,
                p_grad_2_u=self.z_grad_rth_3,
                p_grad_2_v=self.z_grad_rth_4,
                p_ccpr1=self.z_rth_pr_1,
                p_ccpr2=self.z_rth_pr_2,
                geofac_grg_x=self.interpolation_state.geofac_grg_x,
                geofac_grg_y=self.interpolation_state.geofac_grg_y,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),  # UBOUND(p_ccpr,2)
                offset_provider={
                    "C2E2CO": self.grid.get_c2e2co_connectivity(),
                },
            )

        if config.iadv_rhotheta <= 2:
            (tmp_0_0, tmp_0_1) = self.grid.get_indices_from_to(
                EdgeDim,
                HorizontalMarkerIndex.local(EdgeDim) - 2,
                HorizontalMarkerIndex.local(EdgeDim) - 3,
            )
            if config.idiv_method == 1:
                (tmp_0_0, tmp_0_1) = self.grid.get_indices_from_to(
                    EdgeDim,
                    HorizontalMarkerIndex.local(EdgeDim) - 2,
                    HorizontalMarkerIndex.local(EdgeDim) - 2,
                )

            set_zero_e_k.with_backend(run_gtfn)(
                field=z_fields.z_rho_e,
                horizontal_start=tmp_0_0,
                horizontal_end=tmp_0_1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            set_zero_e_k.with_backend(run_gtfn)(
                field=z_fields.z_theta_v_e,
                horizontal_start=tmp_0_0,
                horizontal_end=tmp_0_1,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            # initialize also nest boundary points with zero
            if self.grid.limited_area():
                set_zero_e_k.with_backend(run_gtfn)(
                    field=z_fields.z_rho_e,
                    horizontal_start=indices_4_1,
                    horizontal_end=indices_4_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

                set_zero_e_k.with_backend(run_gtfn)(
                    field=z_fields.z_theta_v_e,
                    horizontal_start=indices_4_1,
                    horizontal_end=indices_4_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

            if config.iadv_rhotheta == 2:
                # Compute upwind-biased values for rho and theta starting from centered differences
                # Note: the length of the backward trajectory should be 0.5*dtime*(vn,vt) in order to arrive
                # at a second-order accurate FV discretization, but twice the length is needed for numerical stability

                nhsolve_prog.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1.with_backend(
                    run_gtfn
                )(
                    p_vn=prognostic_state[nnow].vn,
                    p_vt=diagnostic_state_nh.vt,
                    pos_on_tplane_e_1=self.interpolation_state.pos_on_tplane_e_1,
                    pos_on_tplane_e_2=self.interpolation_state.pos_on_tplane_e_2,
                    primal_normal_cell_1=edge_geometry.primal_normal_cell[0],
                    dual_normal_cell_1=edge_geometry.dual_normal_cell[0],
                    primal_normal_cell_2=edge_geometry.primal_normal_cell[1],
                    dual_normal_cell_2=edge_geometry.dual_normal_cell[1],
                    p_dthalf=p_dthalf,
                    rho_ref_me=self.metric_state_nonhydro.rho_ref_me,
                    theta_ref_me=self.metric_state_nonhydro.theta_ref_me,
                    z_grad_rth_1=self.z_grad_rth_1,
                    z_grad_rth_2=self.z_grad_rth_2,
                    z_grad_rth_3=self.z_grad_rth_3,
                    z_grad_rth_4=self.z_grad_rth_4,
                    z_rth_pr_1=self.z_rth_pr_1,
                    z_rth_pr_2=self.z_rth_pr_2,
                    z_rho_e=z_fields.z_rho_e,
                    z_theta_v_e=z_fields.z_theta_v_e,
                    # p_distv_bary_1=self.p_distv_bary_1,
                    # p_distv_bary_2=self.p_distv_bary_2,
                    horizontal_start=indices_12_1,
                    horizontal_end=indices_12_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={
                        "E2C": self.grid.get_e2c_connectivity(),
                        "E2EC": self.grid.get_e2ec_connectivity(),
                    },
                )

        # Remaining computations at edge points
        mo_solve_nonhydro_stencil_18.with_backend(run_gtfn)(
            inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
            z_exner_ex_pr=self.z_exner_ex_pr,
            z_gradh_exner=z_fields.z_gradh_exner,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.vertical_params.nflatlev,
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )

        if config.igradp_method == 3:
            # horizontal gradient of Exner pressure, including metric correction
            # horizontal gradient of Exner pressure, Taylor-expansion-based reconstruction

            mo_solve_nonhydro_stencil_19.with_backend(run_gtfn)(
                inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
                c_lin_e=self.interpolation_state.c_lin_e,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=self.vertical_params.nflatlev,
                vertical_end=int32(self.vertical_params.nflat_gradp + 1),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                },
            )

            mo_solve_nonhydro_stencil_20.with_backend(run_gtfn)(
                inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
                z_exner_ex_pr=self.z_exner_ex_pr,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                z_dexner_dz_c_1=self.z_dexner_dz_c_1,
                z_dexner_dz_c_2=self.z_dexner_dz_c_2,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=int32(self.vertical_params.nflat_gradp + 1),
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2EC": self.grid.get_e2ec_connectivity(),
                    "Koff": KDim,
                },
            )
        # compute hydrostatically approximated correction term that replaces downward extrapolation
        if config.igradp_method == 3:
            # TODO @tehrengruber: fix power
            mo_solve_nonhydro_stencil_21.with_backend(run_gtfn)(
                theta_v=prognostic_state[nnow].theta_v,
                ikoffset=self.metric_state_nonhydro.vertoffset_gradp,
                zdiff_gradp=self.metric_state_nonhydro.zdiff_gradp,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
                z_hydro_corr=self.z_hydro_corr,
                grav_o_cpd=params.grav_o_cpd,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=self.grid.n_lev() - 1,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C": self.grid.get_e2c_connectivity(),
                    "E2EC": self.grid.get_e2ec_connectivity(),
                    "Koff": KDim,
                },
            )

        self.z_hydro_corr_horizontal = np_as_located_field(EdgeDim)(
            np.asarray(self.z_hydro_corr)[:, 64]
        )

        if config.igradp_method == 3:
            mo_solve_nonhydro_stencil_22.with_backend(run_gtfn)(
                ipeidx_dsl=self.metric_state_nonhydro.ipeidx_dsl,
                pg_exdist=self.metric_state_nonhydro.pg_exdist,
                z_hydro_corr=self.z_hydro_corr_horizontal,
                z_gradh_exner=z_fields.z_gradh_exner,
                horizontal_start=indices_5b_1,
                horizontal_end=indices_5b_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_24.with_backend(run_gtfn)(
            vn_nnow=prognostic_state[nnow].vn,
            ddt_vn_apc_ntl1=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
            ddt_vn_phy=diagnostic_state_nh.ddt_vn_phy,
            z_theta_v_e=z_fields.z_theta_v_e,
            z_gradh_exner=z_fields.z_gradh_exner,
            vn_nnew=prognostic_state[nnew].vn,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if config.is_iau_active:
            mo_solve_nonhydro_stencil_28(
                vn_incr=diagnostic_state_nh.vn_incr,
                vn=prognostic_state[nnew].vn,
                iau_wgt_dyn=config.iau_wgt_dyn,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        if self.grid.limited_area():
            mo_solve_nonhydro_stencil_29.with_backend(run_gtfn)(
                grf_tend_vn=diagnostic_state_nh.grf_tend_vn,
                vn_now=prognostic_state[nnow].vn,
                vn_new=prognostic_state[nnew].vn,
                dtime=dtime,
                horizontal_start=indices_6_1,
                horizontal_end=indices_6_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        mo_solve_nonhydro_stencil_30.with_backend(run_gtfn)(
            e_flx_avg=self.interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            z_vn_avg=self.z_vn_avg,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            vt=diagnostic_state_nh.vt,
            horizontal_start=indices_7_1,
            horizontal_end=indices_7_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                "E2C2E": self.grid.get_e2c2e_connectivity(),
            },
        )

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_32.with_backend(run_gtfn)(
                z_rho_e=z_fields.z_rho_e,
                z_vn_avg=self.z_vn_avg,
                ddqz_z_full_e=self.metric_state_nonhydro.ddqz_z_full_e,
                z_theta_v_e=z_fields.z_theta_v_e,
                mass_fl_e=diagnostic_state_nh.mass_fl_e,
                z_theta_v_fl_e=self.z_theta_v_fl_e,
                horizontal_start=indices_7_1,
                horizontal_end=indices_7_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )
        # TODO @tehrengruber: fix power for 36
        nhsolve_prog.predictor_stencils_35_36.with_backend(run_gtfn)(
            vn=prognostic_state[nnew].vn,
            ddxn_z_full=self.metric_state_nonhydro.ddxn_z_full,
            ddxt_z_full=self.metric_state_nonhydro.ddxt_z_full,
            vt=diagnostic_state_nh.vt,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_e=self.metric_state_nonhydro.wgtfac_e,
            vn_ie=diagnostic_state_nh.vn_ie,
            z_vt_ie=self.z_vt_ie,
            z_kin_hor_e=self.z_kin_hor_e,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_7_1,
            horizontal_end=indices_7_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if not self.l_vert_nested:
            nhsolve_prog.predictor_stencils_37_38.with_backend(run_gtfn)(
                vn=prognostic_state[nnew].vn,
                vt=diagnostic_state_nh.vt,
                vn_ie=diagnostic_state_nh.vn_ie,
                z_vt_ie=self.z_vt_ie,
                z_kin_hor_e=self.z_kin_hor_e,
                wgtfacq_e_dsl=self.metric_state_nonhydro.wgtfacq_e_dsl,
                k_field=self.k_field,
                nlev=self.grid.n_lev(),
                horizontal_start=indices_7_1,
                horizontal_end=indices_7_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={"Koff": KDim},
            )

        nhsolve_prog.stencils_39_40.with_backend(run_gtfn)(
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_w_concorr_me=self.z_w_concorr_me,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            wgtfacq_c_dsl=self.metric_state_nonhydro.wgtfacq_c_dsl,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex_plus1=int32(self.vertical_params.nflatlev + 1),
            nlev=self.grid.n_lev(),
            horizontal_start=indices_9_1,
            horizontal_end=indices_9_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={
                "C2E": self.grid.get_c2e_connectivity(),
                "C2CE": self.grid.get_c2ce_connectivity(),
                "Koff": KDim,
            },
        )

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_41.with_backend(run_gtfn)(
                geofac_div=self.interpolation_state.geofac_div,
                mass_fl_e=diagnostic_state_nh.mass_fl_e,
                z_theta_v_fl_e=self.z_theta_v_fl_e,
                z_flxdiv_mass=self.z_flxdiv_mass,
                z_flxdiv_theta=self.z_flxdiv_theta,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2CE": self.grid.get_c2ce_connectivity(),
                },
            )

        nhsolve_prog.stencils_43_44_45_45b.with_backend(run_gtfn)(
            z_w_expl=z_fields.z_w_expl,
            w_nnow=prognostic_state[nnow].w,
            ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            z_beta=z_fields.z_beta,
            exner_nnow=prognostic_state[nnow].exner,
            rho_nnow=prognostic_state[nnow].rho,
            theta_v_nnow=prognostic_state[nnow].theta_v,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            z_alpha=z_fields.z_alpha,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_q=z_fields.z_q,
            k_field=self.k_field,
            rd=constants.RD,
            cvd=constants.CVD,
            dtime=dtime,
            cpd=constants.CPD,
            nlev=self.grid.n_lev(),
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev() + 1,
            offset_provider={},
        )

        if not (config.l_open_ubc and self.l_vert_nested):
            mo_solve_nonhydro_stencil_46.with_backend(run_gtfn)(
                w_nnew=prognostic_state[nnew].w,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=0,
                vertical_end=1,
                offset_provider={},
            )
        nhsolve_prog.stencils_47_48_49.with_backend(run_gtfn)(
            w_nnew=prognostic_state[nnew].w,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            z_rho_expl=z_fields.z_rho_expl,
            z_exner_expl=z_fields.z_exner_expl,
            rho_nnow=prognostic_state[nnow].rho,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            z_flxdiv_mass=self.z_flxdiv_mass,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_beta=z_fields.z_beta,
            z_flxdiv_theta=self.z_flxdiv_theta,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
            k_field=self.k_field,
            dtime=dtime,
            cell_startindex_nudging_plus1=indices_10_1,
            cell_endindex_interior=indices_10_2,
            nlev=self.grid.n_lev(),
            nlev_k=self.grid.n_lev() + 1,
            offset_provider={
                "Koff": KDim,
            },
        )

        if config.is_iau_active:
            mo_solve_nonhydro_stencil_50.with_backend(run_gtfn)(
                z_fields.z_rho_expl,
                z_fields.z_exner_expl,
                diagnostic_state_nh.rho_incr,
                diagnostic_state_nh.exner_incr,
                config.iau_wgt_dyn,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_52.with_backend(run_gtfn)(
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            z_w_expl=z_fields.z_w_expl,
            z_exner_expl=z_fields.z_exner_expl,
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        mo_solve_nonhydro_stencil_53.with_backend(run_gtfn)(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54.with_backend(run_gtfn)(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=1,
                vertical_end=int32(
                    self.vertical_params.index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_55.with_backend(run_gtfn)(
            z_rho_expl=z_fields.z_rho_expl,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            rho_ic=diagnostic_state_nh.rho_ic,
            w=prognostic_state[nnew].w,
            z_exner_expl=z_fields.z_exner_expl,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            rho_now=prognostic_state[nnow].rho,
            theta_v_now=prognostic_state[nnow].theta_v,
            exner_now=prognostic_state[nnow].exner,
            rho_new=prognostic_state[nnew].rho,
            exner_new=prognostic_state[nnew].exner,
            theta_v_new=prognostic_state[nnew].theta_v,
            dtime=dtime,
            cvd_o_rd=constants.CVD_O_RD,
            horizontal_start=indices_10_1,
            horizontal_end=indices_10_2,
            vertical_start=int32(self.jk_start),
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        # compute dw/dz for divergence damping term
        if config.lhdiff_rcf and config.divdamp_type >= 3:
            mo_solve_nonhydro_stencil_56_63.with_backend(run_gtfn)(
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=indices_10_1,
                horizontal_end=indices_10_2,
                vertical_start=params.kstart_dd3d,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        if idyn_timestep == 1:
            mo_solve_nonhydro_stencil_59.with_backend(run_gtfn)(
                exner=prognostic_state[nnow].exner,
                exner_dyn_incr=self.exner_dyn_incr,
                cell_startindex_nudging_plus1=indices_10_1,
                cell_endindex_interior=indices_10_2,
                kstart_moist=params.kstart_moist,
                nlev=self.grid.n_lev(),
                offset_provider={},
            )

        if self.grid.limited_area():  # for MPI-parallelized case
            nhsolve_prog.stencils_61_62.with_backend(run_gtfn)(
                rho_now=prognostic_state[nnow].rho,
                grf_tend_rho=diagnostic_state_nh.grf_tend_rho,
                theta_v_now=prognostic_state[nnow].theta_v,
                grf_tend_thv=diagnostic_state_nh.grf_tend_thv,
                w_now=prognostic_state[nnow].w,
                grf_tend_w=diagnostic_state_nh.grf_tend_w,
                rho_new=prognostic_state[nnew].rho,
                exner_new=prognostic_state[nnew].exner,
                w_new=prognostic_state[nnew].w,
                k_field=self.k_field,
                dtime=dtime,
                nlev=self.grid.n_lev(),
                horizontal_start=indices_11_1,
                horizontal_end=indices_11_2,
                vertical_start=0,
                vertical_end=int32(self.grid.n_lev() + 1),
                offset_provider={},
            )

        if config.lhdiff_rcf and config.divdamp_type >= 3:
            mo_solve_nonhydro_stencil_56_63.with_backend(run_gtfn)(
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                z_dwdz_dd=z_fields.z_dwdz_dd,
                horizontal_start=indices_11_1,
                horizontal_end=indices_11_2,
                vertical_start=params.kstart_dd3d,
                vertical_end=self.grid.n_lev(),
                offset_provider={"Koff": KDim},
            )

        ##### COMMUNICATION PHASE

    def run_corrector_step(
        self,
        diagnostic_state_nh: DiagnosticStateNonHydro,
        prognostic_state: list[PrognosticState],
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        nh_constants: NHConstants,
        prep_adv: PrepAdvection,
        dtime: float,
        nnew: int,
        nnow: int,
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        lclean_mflx: bool,
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool,
    ):


        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / config.ndyn_substeps_var

        (indices_0_1, indices_0_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
            HorizontalMarkerIndex.local(CellDim),
        )

        (indices_0_3, indices_0_4) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (indices_1_1, indices_1_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.nudging(EdgeDim) + 1,
            HorizontalMarkerIndex.local(EdgeDim),
        )

        (indices_2_1, indices_2_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
            HorizontalMarkerIndex.local(EdgeDim) - 2,
        )

        (indices_3_1, indices_3_2) = self.grid.get_indices_from_to(
            EdgeDim,
            HorizontalMarkerIndex.lateral_boundary(EdgeDim),
            HorizontalMarkerIndex.end(EdgeDim),
        )

        (indices_4_1, indices_4_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.lateral_boundary(CellDim),
            HorizontalMarkerIndex.nudging(CellDim),
        )

        (indices_5_1, indices_5_2) = self.grid.get_indices_from_to(
            CellDim,
            HorizontalMarkerIndex.nudging(CellDim),
            HorizontalMarkerIndex.local(CellDim),
        )

        lvn_only = False
        velocity_advection.VelocityAdvection(
            self.grid,
            self.metric_state_nonhydro,
            self.interpolation_state,
            self.vertical_params
        ).run_corrector_step(
            vn_only=lvn_only,
            diagnostic_state=diagnostic_state_nh,
            prognostic_state=prognostic_state[nnew],
            z_kin_hor_e=z_kin_hor_e,
            z_vt_ie=z_vt_ie,
            inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
            inv_primal_edge_length=edge_geometry.inverse_primal_edge_lengths,
            dtime=dtime,
            ntnd=self.ntl2,
            tangent_orientation=edge_geometry.tangent_orientation,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            cell_areas=cell_areas,
            owner_mask=owner_mask,
            f_e=f_e,
            area_edge=area_edge,
        )

        nvar = nnew

        #  Precompute Rayleigh damping factor
        compute_z_raylfac.with_backend(run_gtfn)(
            self.metric_state_nonhydro.rayleigh_w,
            dtime,
            self.z_raylfac,
            offset_provider={},
        )

        mo_solve_nonhydro_stencil_10.with_backend(run_gtfn)(
            w=prognostic_state[nnew].w,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            rho_now=prognostic_state[nnow].rho,
            rho_var=prognostic_state[nvar].rho,
            theta_now=prognostic_state[nnow].theta_v,
            theta_var=prognostic_state[nvar].theta_v,
            wgtfac_c=self.metric_state_nonhydro.wgtfac_c,
            theta_ref_mc=self.metric_state_nonhydro.theta_ref_mc,
            vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
            exner_pr=diagnostic_state_nh.exner_pr,
            d_exner_dz_ref_ic=self.metric_state_nonhydro.d_exner_dz_ref_ic,
            rho_ic=diagnostic_state_nh.rho_ic,
            z_theta_v_pr_ic=self.z_theta_v_pr_ic,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            z_th_ddz_exner_c=self.z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_rth=nh_constants.wgt_nnow_rth,
            wgt_nnew_rth=nh_constants.wgt_nnew_rth,
            horizontal_start=indices_0_1,
            horizontal_end=indices_0_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if config.l_open_ubc and not self.l_vert_nested:
            raise NotImplementedError("l_open_ubc not implemented")

        mo_solve_nonhydro_stencil_17.with_backend(run_gtfn)(
            hmask_dd3d=self.metric_state_nonhydro.hmask_dd3d,
            scalfac_dd3d=self.metric_state_nonhydro.scalfac_dd3d,
            inv_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
            z_dwdz_dd=z_fields.z_dwdz_dd,
            z_graddiv_vn=z_fields.z_graddiv_vn,
            horizontal_start=indices_0_3,
            horizontal_end=indices_0_4,
            vertical_start=params.kstart_dd3d,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C": self.grid.get_e2c_connectivity(),
            },
        )

        if config.itime_scheme == 4:
            mo_solve_nonhydro_stencil_23.with_backend(run_gtfn)(
                vn_nnow=prognostic_state[nnow].vn,
                ddt_vn_apc_ntl1=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl1],
                ddt_vn_apc_ntl2=diagnostic_state_nh.ddt_vn_apc_pc[self.ntl2],
                ddt_vn_phy=diagnostic_state_nh.ddt_vn_phy,
                z_theta_v_e=z_fields.z_theta_v_e,
                z_gradh_exner=z_fields.z_gradh_exner,
                vn_nnew=prognostic_state[nnew].vn,
                dtime=dtime,
                wgt_nnow_vel=nh_constants.wgt_nnow_vel,
                wgt_nnew_vel=nh_constants.wgt_nnew_vel,
                cpd=constants.CPD,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        if config.lhdiff_rcf and (
            config.divdamp_order == 24 or config.divdamp_order == 4
        ):
            # verified for e-10
            mo_solve_nonhydro_stencil_25.with_backend(run_gtfn)(
                geofac_grdiv=self.interpolation_state.geofac_grdiv,
                z_graddiv_vn=z_fields.z_graddiv_vn,
                z_graddiv2_vn=self.z_graddiv2_vn,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "E2C2EO": self.grid.get_e2c2eo_connectivity(),
                },
            )

        if config.lhdiff_rcf:
            if config.divdamp_order == 24 and nh_constants.scal_divdamp_o2 > 1.0e-6:
                mo_solve_nonhydro_stencil_26.with_backend(run_gtfn)(
                    z_graddiv_vn=z_fields.z_graddiv_vn,
                    vn=prognostic_state[nnew].vn,
                    scal_divdamp_o2=nh_constants.scal_divdamp_o2,
                    horizontal_start=indices_1_1,
                    horizontal_end=indices_1_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

            # TODO: this does not get accessed in FORTRAN
            if (
                config.divdamp_order == 24
                and config.divdamp_fac_o2 <= 4 * config.divdamp_fac
            ):
                if self.grid.limited_area():
                    mo_solve_nonhydro_stencil_27.with_backend(run_gtfn)(
                        scal_divdamp=self.scal_divdamp,
                        bdy_divdamp=bdy_divdamp,
                        nudgecoeff_e=self.interpolation_state.nudgecoeff_e,
                        z_graddiv2_vn=self.z_graddiv2_vn,
                        vn=prognostic_state[nnew].vn,
                        horizontal_start=indices_1_1,
                        horizontal_end=indices_1_2,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )
                else:
                    mo_solve_nonhydro_4th_order_divdamp.with_backend(run_gtfn)(
                        scal_divdamp=self.scal_divdamp,
                        z_graddiv2_vn=self.z_graddiv2_vn,
                        vn=prognostic_state[nnew].vn,
                        horizontal_start=indices_1_1,
                        horizontal_end=indices_1_2,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )

        # TODO: this does not get accessed in FORTRAN
        if config.is_iau_active:
            mo_solve_nonhydro_stencil_28(
                diagnostic_state_nh.vn_incr,
                prognostic_state[nnew].vn,
                config.iau_wgt_dyn,
                horizontal_start=indices_1_1,
                horizontal_end=indices_1_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE

        mo_solve_nonhydro_stencil_31.with_backend(run_gtfn)(
            e_flx_avg=self.interpolation_state.e_flx_avg,
            vn=prognostic_state[nnew].vn,
            z_vn_avg=self.z_vn_avg,
            horizontal_start=indices_2_1,
            horizontal_end=indices_2_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={
                "E2C2EO": self.grid.get_e2c2eo_connectivity(),
            },
        )

        if config.idiv_method == 1:
            mo_solve_nonhydro_stencil_32.with_backend(run_gtfn)(
                z_rho_e=z_fields.z_rho_e,
                z_vn_avg=self.z_vn_avg,
                ddqz_z_full_e=self.metric_state_nonhydro.ddqz_z_full_e,
                z_theta_v_e=z_fields.z_theta_v_e,
                mass_fl_e=diagnostic_state_nh.mass_fl_e,
                z_theta_v_fl_e=self.z_theta_v_fl_e,
                horizontal_start=indices_2_1,
                horizontal_end=indices_2_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

            if lprep_adv:  # Preparations for tracer advection
                if lclean_mflx:
                    mo_solve_nonhydro_stencil_33.with_backend(run_gtfn)(
                        vn_traj=prep_adv.vn_traj,
                        mass_flx_me=prep_adv.mass_flx_me,
                        horizontal_start=indices_3_1,
                        horizontal_end=indices_3_2,
                        vertical_start=0,
                        vertical_end=self.grid.n_lev(),
                        offset_provider={},
                    )

                mo_solve_nonhydro_stencil_34.with_backend(run_gtfn)(
                    z_vn_avg=self.z_vn_avg,
                    mass_fl_e=diagnostic_state_nh.mass_fl_e,
                    vn_traj=prep_adv.vn_traj,
                    mass_flx_me=prep_adv.mass_flx_me,
                    r_nsubsteps=r_nsubsteps,
                    horizontal_start=indices_2_1,
                    horizontal_end=indices_2_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

        if config.idiv_method == 1:
            # verified for e-9
            mo_solve_nonhydro_stencil_41.with_backend(run_gtfn)(
                geofac_div=self.interpolation_state.geofac_div,
                mass_fl_e=diagnostic_state_nh.mass_fl_e,
                z_theta_v_fl_e=self.z_theta_v_fl_e,
                z_flxdiv_mass=self.z_flxdiv_mass,
                z_flxdiv_theta=self.z_flxdiv_theta,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={
                    "C2E": self.grid.get_c2e_connectivity(),
                    "C2CE": self.grid.get_c2ce_connectivity(),
                },
            )

        if config.itime_scheme == 4:
            nhsolve_prog.stencils_42_44_45_45b.with_backend(run_gtfn)(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
                ddt_w_adv_ntl2=diagnostic_state_nh.ddt_w_adv_pc[self.ntl2],
                z_th_ddz_exner_c=self.z_th_ddz_exner_c,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                rho_ic=diagnostic_state_nh.rho_ic,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                z_beta=z_fields.z_beta,
                exner_nnow=prognostic_state[nnow].exner,
                rho_nnow=prognostic_state[nnow].rho,
                theta_v_nnow=prognostic_state[nnow].theta_v,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_alpha=z_fields.z_alpha,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                z_q=z_fields.z_q,
                k_field=self.k_field,
                rd=constants.RD,
                cvd=constants.CVD,
                dtime=dtime,
                cpd=constants.CPD,
                wgt_nnow_vel=nh_constants.wgt_nnow_vel,
                wgt_nnew_vel=nh_constants.wgt_nnew_vel,
                nlev=self.grid.n_lev(),
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={},
            )
        else:
            nhsolve_prog.stencils_43_44_45_45b.with_backend(run_gtfn)(
                z_w_expl=z_fields.z_w_expl,
                w_nnow=prognostic_state[nnow].w,
                ddt_w_adv_ntl1=diagnostic_state_nh.ddt_w_adv_pc[self.ntl1],
                z_th_ddz_exner_c=self.z_th_ddz_exner_c,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                rho_ic=diagnostic_state_nh.rho_ic,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                z_beta=z_fields.z_beta,
                exner_nnow=prognostic_state[nnow].exner,
                rho_nnow=prognostic_state[nnow].rho,
                theta_v_nnow=prognostic_state[nnow].theta_v,
                inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
                z_alpha=z_fields.z_alpha,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                theta_v_ic=diagnostic_state_nh.theta_v_ic,
                z_q=z_fields.z_q,
                k_field=self.k_field,
                rd=constants.RD,
                cvd=constants.CVD,
                dtime=dtime,
                cpd=constants.CPD,
                nlev=self.grid.n_lev(),
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev() + 1,
                offset_provider={},
            )

        if not config.l_open_ubc and not self.l_vert_nested:
            mo_solve_nonhydro_stencil_46.with_backend(run_gtfn)(
                w_nnew=prognostic_state[nnew].w,
                z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=0,
                vertical_end=0,
                offset_provider={},
            )

        nhsolve_prog.stencils_47_48_49.with_backend(run_gtfn)(
            w_nnew=prognostic_state[nnew].w,
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            w_concorr_c=diagnostic_state_nh.w_concorr_c,
            z_rho_expl=z_fields.z_rho_expl,
            z_exner_expl=z_fields.z_exner_expl,
            rho_nnow=prognostic_state[nnow].rho,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            z_flxdiv_mass=self.z_flxdiv_mass,
            exner_pr=diagnostic_state_nh.exner_pr,
            z_beta=z_fields.z_beta,
            z_flxdiv_theta=self.z_flxdiv_theta,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddt_exner_phy=diagnostic_state_nh.ddt_exner_phy,
            k_field=self.k_field,
            dtime=dtime,
            cell_startindex_nudging_plus1=indices_5_1,
            cell_endindex_interior=indices_5_2,
            nlev=self.grid.n_lev(),
            nlev_k=self.grid.n_lev() + 1,
            offset_provider={
                "Koff": KDim,
            },
        )

        # TODO: this is not tested in green line so far
        if config.is_iau_active:
            mo_solve_nonhydro_stencil_50(
                z_rho_expl=z_fields.z_rho_expl,
                z_exner_expl=z_fields.z_exner_expl,
                rho_incr=diagnostic_state_nh.rho_incr,
                exner_incr=diagnostic_state_nh.exner_incr,
                iau_wgt_dyn=config.iau_wgt_dyn,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_52.with_backend(run_gtfn)(
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            theta_v_ic=diagnostic_state_nh.theta_v_ic,
            ddqz_z_half=self.metric_state_nonhydro.ddqz_z_half,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            z_w_expl=z_fields.z_w_expl,
            z_exner_expl=z_fields.z_exner_expl,
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            dtime=dtime,
            cpd=constants.CPD,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        mo_solve_nonhydro_stencil_53.with_backend(run_gtfn)(
            z_q=z_fields.z_q,
            w=prognostic_state[nnew].w,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=1,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if config.rayleigh_type == constants.RAYLEIGH_KLEMP:
            ## ACC w_1 -> p_nh%w
            mo_solve_nonhydro_stencil_54.with_backend(run_gtfn)(
                z_raylfac=self.z_raylfac,
                w_1=prognostic_state[nnew].w_1,
                w=prognostic_state[nnew].w,
                horizontal_start=indices_5_1,
                horizontal_end=indices_5_2,
                vertical_start=1,
                vertical_end=int32(
                    self.vertical_params.index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
                offset_provider={},
            )

        mo_solve_nonhydro_stencil_55.with_backend(run_gtfn)(
            z_rho_expl=z_fields.z_rho_expl,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            inv_ddqz_z_full=self.metric_state_nonhydro.inv_ddqz_z_full,
            rho_ic=diagnostic_state_nh.rho_ic,
            w=prognostic_state[nnew].w,
            z_exner_expl=z_fields.z_exner_expl,
            exner_ref_mc=self.metric_state_nonhydro.exner_ref_mc,
            z_alpha=z_fields.z_alpha,
            z_beta=z_fields.z_beta,
            rho_now=prognostic_state[nnow].rho,
            theta_v_now=prognostic_state[nnow].theta_v,
            exner_now=prognostic_state[nnow].exner,
            rho_new=prognostic_state[nnew].rho,
            exner_new=prognostic_state[nnew].exner,
            theta_v_new=prognostic_state[nnew].theta_v,
            dtime=dtime,
            cvd_o_rd=constants.CVD_O_RD,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=int32(self.jk_start),
            vertical_end=self.grid.n_lev(),
            offset_provider={"Koff": KDim},
        )

        if lprep_adv:
            if lclean_mflx:
                set_zero_c_k.with_backend(run_gtfn)(
                    field=prep_adv.mass_flx_ic,
                    horizontal_start=indices_5_1,
                    horizontal_end=indices_5_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev(),
                    offset_provider={},
                )

        mo_solve_nonhydro_stencil_58.with_backend(run_gtfn)(
            z_contr_w_fl_l=z_fields.z_contr_w_fl_l,
            rho_ic=diagnostic_state_nh.rho_ic,
            vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
            w=prognostic_state[nnew].w,
            mass_flx_ic=prep_adv.mass_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=indices_5_1,
            horizontal_end=indices_5_2,
            vertical_start=0,
            vertical_end=self.grid.n_lev(),
            offset_provider={},
        )

        if lprep_adv:
            if lclean_mflx:
                set_zero_c_k.with_backend(run_gtfn)(
                    field=prep_adv.mass_flx_ic,
                    horizontal_start=indices_4_1,
                    horizontal_end=indices_4_2,
                    vertical_start=0,
                    vertical_end=self.grid.n_lev() + 1,
                    offset_provider={},
                )

            mo_solve_nonhydro_stencil_65.with_backend(run_gtfn)(
                rho_ic=diagnostic_state_nh.rho_ic,
                vwind_expl_wgt=self.metric_state_nonhydro.vwind_expl_wgt,
                vwind_impl_wgt=self.metric_state_nonhydro.vwind_impl_wgt,
                w_now=prognostic_state[nnow].w,
                w_new=prognostic_state[nnew].w,
                w_concorr_c=diagnostic_state_nh.w_concorr_c,
                mass_flx_ic=prep_adv.mass_flx_ic,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=indices_4_1,
                horizontal_end=indices_4_2,
                vertical_start=0,
                vertical_end=self.grid.n_lev(),
                offset_provider={},
            )

        ##### COMMUNICATION PHASE
