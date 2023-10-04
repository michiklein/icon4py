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


from icon4py.model.common.test_utils.fixtures import (  # noqa F401
    backend,
    damping_height,
    data_provider,
    datapath,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    istep,
    jstep,
    linit,
    mesh,
    metrics_savepoint,
    processor_props,
    ranked_data_path,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_exit,
    savepoint_velocity_exit,
    savepoint_velocity_init,
    step_date_exit,
    step_date_init,
    vn_only,
)
from icon4py.model.common.test_utils.pytest_config import (  # noqa: F401
    pytest_addoption,
    pytest_configure,
    pytest_generate_tests,
    pytest_runtest_setup,
)
