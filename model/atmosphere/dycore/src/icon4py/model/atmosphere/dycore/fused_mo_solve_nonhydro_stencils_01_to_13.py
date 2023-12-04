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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, bool, broadcast, int32, maximum, where

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_10 import (
    _mo_solve_nonhydro_stencil_10,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_12 import (
    _mo_solve_nonhydro_stencil_12,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_13 import (
    _mo_solve_nonhydro_stencil_13,
)
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro_program import (
    _predictor_stencils_2_3,
    _predictor_stencils_4_5_6,
    _predictor_stencils_7_8_9,
    _predictor_stencils_11_lower_upper,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _set_zero_c_k
from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _fused_mo_solve_nonhydro_stencils_01_to_13_predictor(
    rho_nnow: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    vert_idx_1d = vert_idx
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    if limited_area:
        (z_rth_pr_1, z_rth_pr_2) = where(
            (horizontal_lower_01 <= horz_idx < horizontal_upper_01),
            (_set_zero_c_k(), _set_zero_c_k()),
            (z_rth_pr_1, z_rth_pr_2),
        )

    (z_exner_ex_pr, exner_pr) = where(
        (horizontal_lower_02 <= horz_idx < horizontal_upper_02) & (vert_idx < (n_lev + int32(1))),
        _predictor_stencils_2_3(
            exner_exfac=exner_exfac,
            exner=exner_nnow,
            exner_ref_mc=exner_ref_mc,
            exner_pr=exner_pr,
            z_exner_ex_pr=z_exner_ex_pr,
            k_field=vert_idx_1d,
            nlev=n_lev,
        ),
        (z_exner_ex_pr, exner_pr),
    )

    vert_start = maximum(1, nflatlev)
    if igradp_method == 3:
        (z_exner_ic, z_dexner_dz_c_1) = where(
            (horizontal_lower_02 <= horz_idx < horizontal_upper_02)
            & (vert_start <= vert_idx < (n_lev + int32(1))),
            _predictor_stencils_4_5_6(
                wgtfacq_c_dsl=wgtfacq_c_dsl,
                z_exner_ex_pr=z_exner_ex_pr,
                z_exner_ic=z_exner_ic,
                wgtfac_c=wgtfac_c,
                inv_ddqz_z_full=inv_ddqz_z_full,
                z_dexner_dz_c_1=z_dexner_dz_c_1,
                k_field=vert_idx_1d,
                nlev=n_lev,
            ),
            (z_exner_ic, z_dexner_dz_c_1),
        )

    (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = where(
        (horizontal_lower_02 <= horz_idx < horizontal_upper_02),
        _predictor_stencils_7_8_9(
            rho=rho_nnow,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v_nnow,
            theta_ref_mc=theta_ref_mc,
            rho_ic=rho_ic,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            wgtfac_c=wgtfac_c,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            k_field=vert_idx_1d,
            nlev=n_lev,
        ),
        (z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )

    # (z_theta_v_pr_ic, theta_v_ic) = where(
    #     (horizontal_lower_02 <= horz_idx < horizontal_upper_02) & (vert_idx < (n_lev + int32(1))),
    #     _predictor_stencils_11_lower_upper(
    #         wgtfacq_c_dsl=wgtfacq_c_dsl,
    #         z_rth_pr=z_rth_pr_2,
    #         theta_ref_ic=theta_ref_ic,
    #         z_theta_v_pr_ic=z_theta_v_pr_ic,
    #         theta_v_ic=theta_v_ic,
    #         k_field=vert_idx_1d,
    #         nlev=n_lev,
    #     ),
    #     (z_theta_v_pr_ic, theta_v_ic),
    # )
    #
    # vert_start = nflat_gradp
    # if igradp_method == 3:
    #     z_dexner_dz_c_2 = where(
    #         (horizontal_lower_02 <= horz_idx < horizontal_upper_02) & (vert_start <= vert_idx),
    #         _mo_solve_nonhydro_stencil_12(
    #             z_theta_v_pr_ic=z_theta_v_pr_ic,
    #             d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
    #             d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
    #             z_rth_pr_2=z_rth_pr_2,
    #         ),
    #         z_dexner_dz_c_2,
    #     )
    #
    # (z_rth_pr_1, z_rth_pr_2) = where(
    #     (horizontal_lower_03 <= horz_idx < horizontal_upper_03),
    #     _mo_solve_nonhydro_stencil_13(
    #         rho=rho_nnow,
    #         rho_ref_mc=rho_ref_mc,
    #         theta_v=theta_v_nnow,
    #         theta_ref_mc=theta_ref_mc,
    #     ),
    #     (z_rth_pr_1, z_rth_pr_2),
    # )

    return (
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
        z_dexner_dz_c_1,
        z_rth_pr_1,
        z_rth_pr_2,
        rho_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        z_dexner_dz_c_2,
    )


# @field_operator
# def _fused_mo_solve_nonhydro_stencils_01_to_13_corrector(
#     w: Field[[CellDim, KDim], float],
#     w_concorr_c: Field[[CellDim, KDim], float],
#     ddqz_z_half: Field[[CellDim, KDim], float],
#     rho_nnow: Field[[CellDim, KDim], float],
#     rho_nvar: Field[[CellDim, KDim], float],
#     theta_v_nnow: Field[[CellDim, KDim], float],
#     theta_v_nvar: Field[[CellDim, KDim], float],
#     wgtfac_c: Field[[CellDim, KDim], float],
#     theta_ref_mc: Field[[CellDim, KDim], float],
#     vwind_expl_wgt: Field[[CellDim], float],
#     exner_pr: Field[[CellDim, KDim], float],
#     d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
#     rho_ic: Field[[CellDim, KDim], float],
#     z_theta_v_pr_ic: Field[[CellDim, KDim], float],
#     theta_v_ic: Field[[CellDim, KDim], float],
#     z_th_ddz_exner_c: Field[[CellDim, KDim], float],
#     dtime: float,
#     wgt_nnow_rth: float,
#     wgt_nnew_rth: float,
#     horz_idx: Field[[CellDim], int32],
#     vert_idx: Field[[KDim], int32],
#     horizontal_lower_11: int32,
#     horizontal_upper_11: int32,
#     n_lev: int32,
# ) -> tuple[
#     Field[[CellDim, KDim], float],
#     Field[[CellDim, KDim], float],
#     Field[[CellDim, KDim], float],
#     Field[[CellDim, KDim], float],
# ]:
#     vert_idx = broadcast(vert_idx, (CellDim, KDim))
#
#     (rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c,) = where(
#         (horizontal_lower_11 <= horz_idx < horizontal_upper_11) & (int32(1) <= vert_idx),
#         _mo_solve_nonhydro_stencil_10(
#             w=w,
#             w_concorr_c=w_concorr_c,
#             ddqz_z_half=ddqz_z_half,
#             rho_now=rho_nnow,
#             rho_var=rho_nvar,
#             theta_now=theta_v_nnow,
#             theta_var=theta_v_nvar,
#             wgtfac_c=wgtfac_c,
#             theta_ref_mc=theta_ref_mc,
#             vwind_expl_wgt=vwind_expl_wgt,
#             exner_pr=exner_pr,
#             d_exner_dz_ref_ic=d_exner_dz_ref_ic,
#             dtime=dtime,
#             wgt_nnow_rth=wgt_nnow_rth,
#             wgt_nnew_rth=wgt_nnew_rth,
#         ),
#         (
#             rho_ic,
#             z_theta_v_pr_ic,
#             theta_v_ic,
#             z_th_ddz_exner_c,
#         ),
#     )
#
#     return (
#         rho_ic,
#         z_theta_v_pr_ic,
#         theta_v_ic,
#         z_th_ddz_exner_c,
#     )


@field_operator
def _fused_mo_solve_nonhydro_stencils_01_to_13(
    rho_nnow: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    istep: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    horizontal_lower_11: int32,
    horizontal_upper_11: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    #if istep == 1:
    (
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
        z_dexner_dz_c_1,
        z_rth_pr_1,
        z_rth_pr_2,
        rho_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        z_dexner_dz_c_2,
    ) = _fused_mo_solve_nonhydro_stencils_01_to_13_predictor(
        rho_nnow=rho_nnow,
        rho_ref_mc=rho_ref_mc,
        theta_v_nnow=theta_v_nnow,
        theta_ref_mc=theta_ref_mc,
        z_rth_pr_1=z_rth_pr_1,
        z_rth_pr_2=z_rth_pr_2,
        z_theta_v_pr_ic=z_theta_v_pr_ic,
        theta_ref_ic=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        wgtfacq_c_dsl=wgtfacq_c_dsl,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        exner_pr=exner_pr,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        z_exner_ic=z_exner_ic,
        exner_exfac=exner_exfac,
        exner_nnow=exner_nnow,
        exner_ref_mc=exner_ref_mc,
        z_exner_ex_pr=z_exner_ex_pr,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        limited_area=limited_area,
        igradp_method=igradp_method,
        horizontal_lower_01=horizontal_lower_01,
        horizontal_upper_01=horizontal_upper_01,
        horizontal_lower_02=horizontal_lower_02,
        horizontal_upper_02=horizontal_upper_02,
        horizontal_lower_03=horizontal_lower_03,
        horizontal_upper_03=horizontal_upper_03,
        n_lev=n_lev,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
    )
    # else:
    #
    #     (
    #         rho_ic,
    #         z_theta_v_pr_ic,
    #         theta_v_ic,
    #         z_th_ddz_exner_c,
    #     ) = _fused_mo_solve_nonhydro_stencils_01_to_13_corrector(
    #         w=w,
    #         w_concorr_c=w_concorr_c,
    #         ddqz_z_half=ddqz_z_half,
    #         rho_nnow=rho_nnow,
    #         rho_nvar=rho_nvar,
    #         theta_v_nnow=theta_v_nnow,
    #         theta_v_nvar=theta_v_nvar,
    #         wgtfac_c=wgtfac_c,
    #         theta_ref_mc=theta_ref_mc,
    #         vwind_expl_wgt=vwind_expl_wgt,
    #         exner_pr=exner_pr,
    #         d_exner_dz_ref_ic=d_exner_dz_ref_ic,
    #         rho_ic=rho_ic,
    #         z_theta_v_pr_ic=z_theta_v_pr_ic,
    #         theta_v_ic=theta_v_ic,
    #         z_th_ddz_exner_c=z_th_ddz_exner_c,
    #         dtime=dtime,
    #         wgt_nnow_rth=wgt_nnow_rth,
    #         wgt_nnew_rth=wgt_nnew_rth,
    #         horz_idx=horz_idx,
    #         vert_idx=vert_idx,
    #         horizontal_lower_11=horizontal_lower_11,
    #         horizontal_upper_11=horizontal_upper_11,
    #         n_lev=n_lev,
    #     )

    return (
        z_exner_ex_pr,
        exner_pr,
        z_exner_ic,
        z_dexner_dz_c_1,
        z_rth_pr_1,
        z_rth_pr_2,
        rho_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        z_dexner_dz_c_2,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def fused_mo_solve_nonhydro_stencils_01_to_13(
    rho_nnow: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    exner_exfac: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    limited_area: bool,
    igradp_method: int32,
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    rho_nvar: Field[[CellDim, KDim], float],
    theta_v_nvar: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    istep: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_02: int32,
    horizontal_upper_02: int32,
    horizontal_lower_03: int32,
    horizontal_upper_03: int32,
    horizontal_lower_11: int32,
    horizontal_upper_11: int32,
    n_lev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
):
    _fused_mo_solve_nonhydro_stencils_01_to_13(
        rho_nnow=rho_nnow,
        rho_ref_mc=rho_ref_mc,
        theta_v_nnow=theta_v_nnow,
        theta_ref_mc=theta_ref_mc,
        z_rth_pr_1=z_rth_pr_1,
        z_rth_pr_2=z_rth_pr_2,
        z_theta_v_pr_ic=z_theta_v_pr_ic,
        theta_ref_ic=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        wgtfacq_c_dsl=wgtfacq_c_dsl,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        exner_pr=exner_pr,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        z_exner_ic=z_exner_ic,
        exner_exfac=exner_exfac,
        exner_nnow=exner_nnow,
        exner_ref_mc=exner_ref_mc,
        z_exner_ex_pr=z_exner_ex_pr,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        limited_area=limited_area,
        igradp_method=igradp_method,
        w=w,
        w_concorr_c=w_concorr_c,
        rho_nvar=rho_nvar,
        theta_v_nvar=theta_v_nvar,
        dtime=dtime,
        wgt_nnow_rth=wgt_nnow_rth,
        wgt_nnew_rth=wgt_nnew_rth,
        istep=istep,
        horizontal_lower_01=horizontal_lower_01,
        horizontal_upper_01=horizontal_upper_01,
        horizontal_lower_02=horizontal_lower_02,
        horizontal_upper_02=horizontal_upper_02,
        horizontal_lower_03=horizontal_lower_03,
        horizontal_upper_03=horizontal_upper_03,
        horizontal_lower_11=horizontal_lower_11,
        horizontal_upper_11=horizontal_upper_11,
        n_lev=n_lev,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        out=(
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
            z_dexner_dz_c_1,
            z_rth_pr_1,
            z_rth_pr_2,
            rho_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
            z_dexner_dz_c_2,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )

    _fused_mo_solve_nonhydro_stencils_01_to_13(
        rho_nnow=rho_nnow,
        rho_ref_mc=rho_ref_mc,
        theta_v_nnow=theta_v_nnow,
        theta_ref_mc=theta_ref_mc,
        z_rth_pr_1=z_rth_pr_1,
        z_rth_pr_2=z_rth_pr_2,
        z_theta_v_pr_ic=z_theta_v_pr_ic,
        theta_ref_ic=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        wgtfacq_c_dsl=wgtfacq_c_dsl,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        exner_pr=exner_pr,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        z_exner_ic=z_exner_ic,
        exner_exfac=exner_exfac,
        exner_nnow=exner_nnow,
        exner_ref_mc=exner_ref_mc,
        z_exner_ex_pr=z_exner_ex_pr,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        limited_area=limited_area,
        igradp_method=igradp_method,
        w=w,
        w_concorr_c=w_concorr_c,
        rho_nvar=rho_nvar,
        theta_v_nvar=theta_v_nvar,
        dtime=dtime,
        wgt_nnow_rth=wgt_nnow_rth,
        wgt_nnew_rth=wgt_nnew_rth,
        istep=istep,
        horizontal_lower_01=horizontal_lower_01,
        horizontal_upper_01=horizontal_upper_01,
        horizontal_lower_02=horizontal_lower_02,
        horizontal_upper_02=horizontal_upper_02,
        horizontal_lower_03=horizontal_lower_03,
        horizontal_upper_03=horizontal_upper_03,
        horizontal_lower_11=horizontal_lower_11,
        horizontal_upper_11=horizontal_upper_11,
        n_lev=n_lev,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        out=(
            z_exner_ex_pr,
            exner_pr,
            z_exner_ic,
            z_dexner_dz_c_1,
            z_rth_pr_1,
            z_rth_pr_2,
            rho_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
            z_dexner_dz_c_2,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )
