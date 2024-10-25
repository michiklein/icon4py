# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.math import operators as math_oper
from icon4py.model.common.settings import xp


def compute_smooth_topo(
    topography: fa.CellField[ta.wpfloat],
    grid: icon_grid.IconGrid,
    cell_areas: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    backend,
    num_iterations: int = 25,
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the smoothed (laplacian-filtered) topography needed by the SLEVE
    coordinate.
    """

    topography_smoothed_np = xp.zeros((grid.num_cells, grid.num_levels), dtype=ta.wpfloat)
    topography_smoothed_np[:, 0] = topography.asnumpy()
    topography_smoothed = gtx.as_field((dims.CellDim, dims.KDim), topography_smoothed_np)

    nabla2_topo_np = xp.zeros((grid.num_cells, grid.num_levels), dtype=ta.wpfloat)
    nabla2_topo = gtx.as_field((dims.CellDim, dims.KDim), nabla2_topo_np)

    cell_domain = h_grid.domain(dims.CellDim)
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    for iter in range(num_iterations):
        math_oper.nabla2_scalar.with_backend(backend)(
            psi_c=topography_smoothed,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_topo,
            horizontal_start=0,
            horizontal_end=end_cell_end,
            vertical_start=0,
            vertical_end=1,
            offset_provider={
                "C2E2CO": grid.get_offset_provider("C2E2CO"),
            },
        )

        topography_smoothed_np[:, 0] = (
            topography_smoothed.asnumpy()[:, 0]
            + 0.125 * nabla2_topo.asnumpy()[:, 0] * cell_areas.asnumpy()
        )
        topography_smoothed = gtx.as_field((dims.CellDim, dims.KDim), topography_smoothed_np)

    return gtx.as_field((dims.CellDim,), topography_smoothed_np[:, 0])
