# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple
from icon4py.model.common.math import helpers
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers as test_helpers


def test_cross_product(backend):
    mesh = simple.SimpleGrid()
    x1 = data_alloc.random_field(mesh, dims.EdgeDim)
    y1 = data_alloc.random_field(mesh, dims.EdgeDim)
    z1 = data_alloc.random_field(mesh, dims.EdgeDim)
    x2 = data_alloc.random_field(mesh, dims.EdgeDim)
    y2 = data_alloc.random_field(mesh, dims.EdgeDim)
    z2 = data_alloc.random_field(mesh, dims.EdgeDim)
    x = data_alloc.zero_field(mesh, dims.EdgeDim)
    y = data_alloc.zero_field(mesh, dims.EdgeDim)
    z = data_alloc.zero_field(mesh, dims.EdgeDim)

    helpers.cross_product_on_edges.with_backend(backend)(
        x1, x2, y1, y2, z1, z2, out=(x, y, z), offset_provider={}
    )
    a = np.column_stack((x1.ndarray, y1.ndarray, z1.ndarray))
    b = np.column_stack((x2.ndarray, y2.ndarray, z2.ndarray))
    c = np.cross(a, b)

    assert test_helpers.dallclose(c[:, 0], x.ndarray)
    assert test_helpers.dallclose(c[:, 1], y.ndarray)
    assert test_helpers.dallclose(c[:, 2], z.ndarray)
