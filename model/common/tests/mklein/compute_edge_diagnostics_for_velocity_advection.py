# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    _compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_to_half_levels(
    wgtfac_e: fa.EdgeKField[ta.wpfloat],
    x: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    x_ie_wp = wgtfac_e * x + (wpfloat("1.0") - wgtfac_e) * x(Koff[-1])
    return concat_where(dims.KDim > 0, x_ie_wp, x)


@gtx.field_operator
def _compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    z_kin_hor_e = wpfloat("0.5") * (vn * vn + vt * vt)
    return z_kin_hor_e


@gtx.field_operator
def _compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    tangential_wind_vp = _compute_tangential_wind(vn, rbf_vec_coeff_e)
    tangential_wind = astype(tangential_wind_vp, wpfloat)
    horizontal_kinetic_energy_at_edges_on_model_levels = _compute_horizontal_kinetic_energy(
        vn, tangential_wind
    )
    vn_on_half_levels = _interpolate_to_half_levels(wgtfac_e, vn)

    tangential_wind_on_half_levels = (
        _interpolate_to_half_levels(wgtfac_e, tangential_wind)
        if not skip_compute_predictor_vertical_advection
        else tangential_wind_on_half_levels
    )

    contravariant_correction_vp = _compute_contravariant_correction(
        vn,
        astype(ddxn_z_full, vpfloat),
        astype(ddxt_z_full, vpfloat),
        astype(tangential_wind, vpfloat),
    )
    contravariant_correction_at_edges_on_model_levels = concat_where(
        nflatlev <= dims.KDim,
        astype(contravariant_correction_vp, wpfloat),
        contravariant_correction_at_edges_on_model_levels,
    )

    w_at_vertices_vp = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)
    w_at_vertices = astype(w_at_vertices_vp, wpfloat)
    horizontal_advection_of_w_vp = _compute_horizontal_advection_term_for_vertical_velocity(
        vn_on_half_levels,
        inv_dual_edge_length,
        w,
        tangential_wind_on_half_levels,
        inv_primal_edge_length,
        tangent_orientation,
        w_at_vertices,
    )
    horizontal_advection_of_w_at_edges_on_half_levels = (
        astype(horizontal_advection_of_w_vp, wpfloat)
        if not skip_compute_predictor_vertical_advection
        else horizontal_advection_of_w_at_edges_on_half_levels
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
    )


@gtx.field_operator
def _compute_horizontal_advection_of_w(
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.wpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    w_at_vertices_vp = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)
    w_at_vertices = astype(w_at_vertices_vp, wpfloat)

    horizontal_advection_of_w_vp = _compute_horizontal_advection_term_for_vertical_velocity(
        vn_on_half_levels,
        inv_dual_edge_length,
        w,
        tangential_wind_on_half_levels,
        inv_primal_edge_length,
        tangent_orientation,
        w_at_vertices,
    )

    return astype(horizontal_advection_of_w_vp, wpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
    tangential_wind: fa.EdgeKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    num_edges: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    

    _compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
        tangential_wind_on_half_levels,
        contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
        vn,
        w,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        c_intp,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        skip_compute_predictor_vertical_advection,
        nflatlev,
        out=(
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_advection_of_w(
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.wpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    num_edges: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    

    _compute_horizontal_advection_of_w(
        w,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        c_intp,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        out=horizontal_advection_of_w_at_edges_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


if __name__ == "__main__":
    

    import argparse
    import time
    import numpy as np
    import cupy as cp

    from mklein_test import (
        get_torus_grid,
        trim_grid,
        reindex_cells,
        reindex_edges,
        reindex_vertices,
        sort_into_blocks_32,
        ToZeroBasedIndexTransformation,
    )
    from icon4py.model.common.dimension import EdgeDim, CellDim, VertexDim, KDim
    from gt4py.next import int32

    b_end = gtx.gtfn_gpu
    xp = cp if "gpu" in str(b_end).lower() else np

    parser = argparse.ArgumentParser(
        description="Run compute_edge_diagnostics_for_velocity_advection with optional block sorting",
    )
    parser.add_argument(
        "--block-sort",
        action="store_true",
        help="Sort edges and cells into blocks of 32 (for *_on_32 job-scripts)",
    )
    args = parser.parse_args()

    grid_file = "../all_torus_files/torus_100000_100000_256_reordered.nc"
    levels = 80

    grid = get_torus_grid(grid_file, levels, ToZeroBasedIndexTransformation())
    vertices, edges, cells = trim_grid(grid_file, grid)

    if args.block_sort:
        edges, cells = sort_into_blocks_32(grid, grid_file, edges, cells)
        reindex_vertices(grid, vertices)
    else:
        reindex_cells(grid, cells)
        reindex_edges(grid, edges)
        reindex_vertices(grid, vertices)

    if xp.__name__ == "cupy":
        cp.get_default_memory_pool().free_all_blocks()
        for name, connectivity in grid.connectivities.items():
            if hasattr(connectivity, "ndarray"):
                connectivity.ndarray = cp.asarray(connectivity.ndarray)
            else:
                grid.connectivities[name] = cp.asarray(connectivity)
        
        # Convert offset providers for GPU
        for name, provider in grid.offset_providers.items():
            if hasattr(provider, 'ndarray'):
                grid.offset_providers[name] = gtx.as_connectivity(
                    provider.domain,
                    codomain=provider.codomain, 
                    data=provider.ndarray, 
                    skip_value=None,
                    allocator=b_end
                )

    rng = np.random.default_rng(1)

    # Determine local connectivity lengths once so that field shapes match the grid
    e2c2e_len = grid.get_offset_provider("E2C2E").ndarray.shape[1]
    v2c_len = grid.get_offset_provider("V2C").ndarray.shape[1]

    # ---------------------------- Allocate input arrays ---------------------------
    vn_arr = xp.asarray(rng.random((grid.num_edges, levels)))
    w_arr = xp.asarray(rng.random((grid.num_cells, levels)))
    rbf_vec_coeff_e_arr = xp.asarray(rng.random((grid.num_edges, e2c2e_len)))
    wgtfac_e_arr = xp.asarray(rng.random((grid.num_edges, levels)))
    ddxn_z_full_arr = xp.asarray(rng.random((grid.num_edges, levels)))
    ddxt_z_full_arr = xp.asarray(rng.random((grid.num_edges, levels)))
    wgtfacq_e_arr = xp.asarray(rng.random((grid.num_edges, levels)))
    c_intp_arr = xp.asarray(rng.random((grid.num_vertices, v2c_len)))
    inv_dual_edge_length_arr = xp.asarray(rng.random((grid.num_edges,)))
    inv_primal_edge_length_arr = xp.asarray(rng.random((grid.num_edges,)))
    tangent_orientation_arr = xp.asarray(rng.random((grid.num_edges,)))

    # ----------------------------- Create GT4Py fields ---------------------------
    edge_k_domain = gtx.domain({EdgeDim: grid.num_edges, KDim: levels})
    cell_k_domain = gtx.domain({CellDim: grid.num_cells, KDim: levels})
    edge_domain = gtx.domain({EdgeDim: grid.num_edges})
    vertex_domain = gtx.domain({VertexDim: grid.num_vertices, dims.V2CDim: v2c_len})
    e2c2e_field_domain = gtx.domain({EdgeDim: grid.num_edges, dims.E2C2EDim: e2c2e_len})

    vn = gtx.as_field(edge_k_domain, vn_arr, allocator=b_end)
    w = gtx.as_field(cell_k_domain, w_arr, allocator=b_end)
    rbf_vec_coeff_e = gtx.as_field(e2c2e_field_domain, rbf_vec_coeff_e_arr, allocator=b_end)
    wgtfac_e = gtx.as_field(edge_k_domain, wgtfac_e_arr, allocator=b_end)
    ddxn_z_full = gtx.as_field(edge_k_domain, ddxn_z_full_arr, allocator=b_end)
    ddxt_z_full = gtx.as_field(edge_k_domain, ddxt_z_full_arr, allocator=b_end)
    wgtfacq_e = gtx.as_field(edge_k_domain, wgtfacq_e_arr, allocator=b_end)
    c_intp = gtx.as_field(vertex_domain, c_intp_arr, allocator=b_end)
    inv_dual_edge_length = gtx.as_field(edge_domain, inv_dual_edge_length_arr, allocator=b_end)
    inv_primal_edge_length = gtx.as_field(edge_domain, inv_primal_edge_length_arr, allocator=b_end)
    tangent_orientation = gtx.as_field(edge_domain, tangent_orientation_arr, allocator=b_end)

    # ----------------------------- Output fields --------------------------------
    tangential_wind = gtx.zeros(edge_k_domain, allocator=b_end)
    tangential_wind_on_half_levels = gtx.zeros(edge_k_domain, allocator=b_end)
    vn_on_half_levels = gtx.zeros(edge_k_domain, allocator=b_end)
    horizontal_kinetic_energy = gtx.zeros(edge_k_domain, allocator=b_end)
    contravariant_correction = gtx.zeros(edge_k_domain, allocator=b_end)
    horizontal_advection_of_w = gtx.zeros(edge_k_domain, allocator=b_end)

    horizontal_start = int32(0)
    horizontal_end = int32(grid.num_edges)
    num_edges = int32(grid.num_edges)
    vertical_start = int32(0)
    vertical_end = int32(levels)

    print("start")
    for _ in range(1000):
        compute_horizontal_advection_of_w.with_backend(b_end)(
            horizontal_advection_of_w,
            w,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            c_intp,
            inv_dual_edge_length,
            inv_primal_edge_length,
            tangent_orientation,
            num_edges,
            horizontal_start,
            horizontal_end,
            vertical_start,
            vertical_end,
            offset_provider=grid.offset_providers,
        )
    print("end")
