# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


INTERFACE_LEVEL_HEIGHT_STANDARD_NAME: Final[str] = "model_interface_height"
INTERFACE_LEVEL_STANDARD_NAME: Final[str] = "interface_model_level_number"

attrs: Final[dict[str, model.FieldMetaData]] = {
    "theta_ref_mc": dict(
        standard_name="theta_ref_mc",
        long_name="theta_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="theta_ref_mc",
        dtype=ta.wpfloat,
    ),
    "exner_ref_mc": dict(
        standard_name="exner_ref_mc",
        long_name="exner_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="exner_ref_mc",
        dtype=ta.wpfloat,
    ),
    "z_ifv": dict(
        standard_name="z_ifv",
        long_name="z_ifv",
        units="",
        dims=(dims.VertexDim, dims.KDim),
        icon_var_name="z_ifv",
        dtype=ta.wpfloat,
    ),
    "vert_out": dict(
        standard_name="vert_out",
        long_name="vert_out",
        units="",
        dims=(dims.VertexDim, dims.KDim),
        icon_var_name="vert_out",
        dtype=ta.wpfloat,
    ),
    "functional_determinant_of_metrics_on_interface_levels": dict(
        standard_name="functional_determinant_of_metrics_on_interface_levels",
        long_name="functional determinant of the metrics [sqrt(gamma)] on half levels",
        units="",
        dims=(dims.CellDim, dims.KHalfDim),
        dtype=ta.wpfloat,
        icon_var_name="ddqz_z_half",
    ),
    "height": dict(
        standard_name="height",
        long_name="height",
        units="m",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_mc",
        dtype=ta.wpfloat,
    ),
    "height_on_interface_levels": dict(
        standard_name="height_on_interface_levels",
        long_name="height_on_interface_levels",
        units="m",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="z_ifc",
        dtype=ta.wpfloat,
    ),
    "z_ifc_sliced": dict(
        standard_name="z_ifc_sliced",
        long_name="z_ifc_sliced",
        units="m",
        dims=(dims.CellDim),
        icon_var_name="z_ifc_sliced",
        dtype=ta.wpfloat,
    ),
    "model_level_number": dict(
        standard_name="model_level_number",
        long_name="model level number",
        units="",
        dims=(dims.KDim,),
        icon_var_name="k_index",
        dtype=gtx.int32,
    ),
    INTERFACE_LEVEL_STANDARD_NAME: dict(
        standard_name=INTERFACE_LEVEL_STANDARD_NAME,
        long_name="model interface level number",
        units="",
        dims=(dims.KHalfDim,),
        icon_var_name="k_index",
        dtype=gtx.int32,
    ),
    "weighting_factor_for_quadratic_interpolation_to_cell_surface": dict(
        standard_name="weighting_factor_for_quadratic_interpolation_to_cell_surface",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="wgtfacq_c_dsl",
        long_name="weighting factor for quadratic interpolation to cell surface",
    ),
    "weighting_factor_for_quadratic_interpolation_to_edge_center": dict(
        standard_name="weighting_factor_for_quadratic_interpolation_to_edge_center",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="wgtfacq_e_dsl",
        long_name="weighting factor for quadratic interpolation to edge centers",
    ),
    "cell_to_edge_interpolation_coefficient": dict(
        standard_name="cell_to_edge_interpolation_coefficient",
        units="",
        dims=(dims.EdgeDim, dims.E2CDim),
        dtype=ta.wpfloat,
        icon_var_name="c_lin_e",
        long_name="coefficients for cell to edge interpolation",
    ),
    "scaling_factor_for_3d_divergence_damping": dict(
        standard_name="scaling_factor_for_3d_divergence_damping",
        units="",
        dims=(dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="scalfac_dd3d",
        long_name="Scaling factor for 3D divergence damping terms",
    ),
    "model_interface_height": dict(
        standard_name="model_interface_height",
        long_name="height value of half levels without topography",
        units="m",
        dims=(dims.KHalfDim,),
        dtype=ta.wpfloat,
        positive="up",
        icon_var_name="vct_a",
    ),
    "nudging_coefficient_on_edges": dict(
        standard_name="nudging_coefficient_on_edges",
        long_name="nudging coefficients on edges",
        units="",
        dtype=ta.wpfloat,
        dims=(dims.EdgeDim,),
        icon_var_name="nudgecoeff_e",
    ),
    "refin_e_ctrl": dict(
        standard_name="refin_e_ctrl",
        long_name="grid refinement control on edgeds",
        units="",
        dtype=int,
        dims=(dims.EdgeDim,),
        icon_var_name="refin_e_ctrl",
    ),
    "c_bln_avg": dict(
        standard_name="c_bln_avg",
        units="",
        dims=(dims.CellDim, dims.C2E2CODim),
        dtype=ta.wpfloat,
        icon_var_name="c_bln_avg",
        long_name="grid savepoint field",
    ),
    "vct_a": dict(
        standard_name="vct_a",
        units="",
        dims=(dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="vct_a",
        long_name="grid savepoint field",
    ),
    "c_refin_ctrl": dict(
        standard_name="c_refin_ctrl",
        units="",
        dims=(dims.CellDim),
        dtype=ta.wpfloat,
        icon_var_name="c_refin_ctrl",
        long_name="grid savepoint field",
    ),
    "e_refin_ctrl": dict(
        standard_name="e_refin_ctrl",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="e_refin_ctrl",
        long_name="grid savepoint field",
    ),
    "dual_edge_length": dict(
        standard_name="dual_edge_length",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="dual_edge_length",
        long_name="grid savepoint field",
    ),
    "tangent_orientation": dict(
        standard_name="tangent_orientation",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="tangent_orientation",
        long_name="grid savepoint field",
    ),
    "inv_primal_edge_length": dict(
        standard_name="inv_primal_edge_length",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="inv_primal_edge_length",
        long_name="grid savepoint field",
    ),
    "inv_dual_edge_length": dict(
        standard_name="inv_dual_edge_length",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="inv_dual_edge_length",
        long_name="grid savepoint field",
    ),
    "cells_aw_verts_field": dict(
        standard_name="cells_aw_verts_field",
        units="",
        dims=(dims.VertexDim, dims.V2CDim),
        dtype=ta.wpfloat,
        icon_var_name="cells_aw_verts_field",
        long_name="grid savepoint field",
    ),
    "e_lev": dict(
        standard_name="e_lev",
        long_name="e_lev",
        units="",
        dims=(dims.EdgeDim,),
        icon_var_name="e_lev",
        dtype=gtx.int32,
    ),
    "e_owner_mask": dict(
        standard_name="e_owner_mask",
        units="",
        dims=(dims.EdgeDim),
        dtype=bool,
        icon_var_name="e_owner_mask",
        long_name="grid savepoint field",
    ),
    "c_owner_mask": dict(
        standard_name="c_owner_mask",
        units="",
        dims=(dims.CellDim),
        dtype=bool,
        icon_var_name="c_owner_mask",
        long_name="grid savepoint field",
    ),
    "edge_cell_length": dict(
        standard_name="edge_cell_length",
        units="",
        dims=(dims.EdgeDim, dims.E2CDim),
        dtype=ta.wpfloat,
        icon_var_name="edge_cell_length",
        long_name="grid savepoint field",
    ),
    "ddqz_z_full": dict(
        standard_name="ddqz_z_full",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="ddqz_z_full",
        long_name="metrics field",
    ),
    "inv_ddqz_z_full": dict(
        standard_name="inv_ddqz_z_full",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="inv_ddqz_z_full",
        long_name="metrics field",
    ),
    "scalfac_dd3d": dict(
        standard_name="scalfac_dd3d",
        units="",
        dims=(dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="scalfac_dd3d",
        long_name="metrics field",
    ),
    "rayleigh_w": dict(
        standard_name="rayleigh_w",
        units="",
        dims=(dims.KHalfDim),
        dtype=ta.wpfloat,
        icon_var_name="rayleigh_w",
        long_name="metrics field",
    ),
    "coeff1_dwdz": dict(
        standard_name="coeff1_dwdz",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="coeff1_dwdz",
        long_name="metrics field",
    ),
    "coeff2_dwdz": dict(
        standard_name="coeff2_dwdz",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="coeff2_dwdz",
        long_name="metrics field",
    ),
    "d2dexdz2_fac1_mc": dict(
        standard_name="d2dexdz2_fac1_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="d2dexdz2_fac1_mc",
        long_name="metrics field",
    ),
    "d2dexdz2_fac2_mc": dict(
        standard_name="d2dexdz2_fac2_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="d2dexdz2_fac2_mc",
        long_name="metrics field",
    ),
    "ddxt_z_half_e": dict(
        standard_name="ddxt_z_half_e",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="ddxt_z_half_e",
        long_name="metrics field",
    ),
    "ddxn_z_full": dict(
        standard_name="ddxn_z_full",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="ddxn_z_full",
        long_name="metrics field",
    ),
    "ddxn_z_half_e": dict(
        standard_name="ddxn_z_half_e",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="ddxn_z_half_e",
        long_name="metrics field",
    ),
    "vwind_impl_wgt": dict(
        standard_name="vwind_impl_wgt",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="vwind_impl_wgt",
        long_name="metrics field",
    ),
    "vwind_expl_wgt": dict(
        standard_name="vwind_expl_wgt",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="vwind_expl_wgt",
        long_name="metrics field",
    ),
    "exner_exfac": dict(
        standard_name="exner_exfac",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="exner_exfac",
        long_name="metrics field",
    ),
    "z_aux2": dict(
        standard_name="z_aux2",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="z_aux2",
        long_name="metrics field",
    ),
    "flat_idx_max": dict(
        standard_name="flat_idx_max",
        units="",
        dims=(dims.EdgeDim),
        dtype=gtx.int32,
        icon_var_name="flat_idx_max",
        long_name="metrics field",
    ),
    "z_me": dict(
        standard_name="z_me",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="z_me",
        long_name="metrics field",
    ),
    "pg_edgeidx": dict(
        standard_name="pg_edgeidx",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=gtx.int32,
        icon_var_name="pg_edgeidx",
        long_name="metrics field",
    ),
    "pg_vertidx": dict(
        standard_name="pg_vertidx",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=gtx.int32,
        icon_var_name="pg_vertidx",
        long_name="metrics field",
    ),
    "pg_edgeidx_dsl": dict(
        standard_name="pg_edgeidx_dsl",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=bool,
        icon_var_name="pg_edgeidx_dsl",
        long_name="metrics field",
    ),
    "pg_exdist_dsl": dict(
        standard_name="pg_exdist_dsl",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="pg_exdist_dsl",
        long_name="metrics field",
    ),
    "bdy_halo_c": dict(
        standard_name="bdy_halo_c",
        units="",
        dims=(dims.CellDim),
        dtype=bool,
        icon_var_name="bdy_halo_c",
        long_name="metrics field",
    ),
    "hmask_dd3d": dict(
        standard_name="hmask_dd3d",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="hmask_dd3d",
        long_name="metrics field",
    ),
    "zdiff_gradp": dict(
        standard_name="zdiff_gradp",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="zdiff_gradp",
        long_name="metrics field",
    ),
    "coeff_gradekin": dict(
        standard_name="coeff_gradekin",
        units="",
        dims=(dims.EdgeDim),
        dtype=ta.wpfloat,
        icon_var_name="coeff_gradekin",
        long_name="metrics field",
    ),
    "mask_prog_halo_c": dict(
        standard_name="mask_prog_halo_c",
        units="",
        dims=(dims.CellDim),
        dtype=bool,
        icon_var_name="mask_prog_halo_c",
        long_name="metrics field",
    ),
    "mask_hdiff": dict(
        standard_name="mask_hdiff",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=bool,
        icon_var_name="mask_hdiff",
        long_name="metrics field",
    ),
    "max_nbhgt": dict(
        standard_name="max_nbhgt",
        units="",
        dims=(dims.CellDim),
        dtype=ta.wpfloat,
        icon_var_name="max_nbhgt",
        long_name="metrics field",
    ),
    "maxslp": dict(
        standard_name="maxslp",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="maxslp",
        long_name="metrics field",
    ),
    "maxhgtd": dict(
        standard_name="maxhgtd",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="maxhgtd",
        long_name="metrics field",
    ),
    "z_maxslp_avg": dict(
        standard_name="z_maxslp_avg",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="z_maxslp_avg",
        long_name="metrics field",
    ),
    "z_maxhgtd_avg": dict(
        standard_name="z_maxhgtd_avg",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="z_maxhgtd_avg",
        long_name="metrics field",
    ),
    "zd_diffcoef_dsl": dict(
        standard_name="zd_diffcoef_dsl",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="zd_diffcoef_dsl",
        long_name="metrics field",
    ),
    "zd_vertoffset_dsl": dict(
        standard_name="zd_vertoffset_dsl",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="zd_vertoffset_dsl",
        long_name="metrics field",
    ),
    "zd_intcoef_dsl": dict(
        standard_name="zd_intcoef_dsl",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="zd_intcoef_dsl",
        long_name="metrics field",
    ),
}
