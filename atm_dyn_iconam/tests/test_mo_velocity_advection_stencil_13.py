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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_13 import (
    mo_velocity_advection_stencil_13,
)
from icon4py.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field
from .test_utils.simple_mesh import SimpleMesh


def mo_velocity_advection_stencil_13_numpy(
    w_concorr_c: np.array,
    z_w_con_c: np.array,
) -> np.array:
    z_w_con_c = z_w_con_c - w_concorr_c
    return z_w_con_c


def test_mo_velocity_advection_stencil_13():
    mesh = SimpleMesh()

    z_w_con_c = random_field(mesh, CellDim, KDim)
    w_concorr_c = random_field(mesh, CellDim, KDim)

    ref = mo_velocity_advection_stencil_13_numpy(
        np.asarray(w_concorr_c), np.asarray(z_w_con_c)
    )
    mo_velocity_advection_stencil_13(
        w_concorr_c,
        z_w_con_c,
        offset_provider={},
    )
    assert np.allclose(z_w_con_c, ref)
