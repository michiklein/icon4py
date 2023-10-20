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

import pytest
from datetime import datetime
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config

from icon4py.model.common.test_utils.fixtures import (  # noqa: F401
    data_provider,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    datapath,
    download_ser_data,
    processor_props,
    ranked_data_path,
    ndyn_substeps,
    damping_height,
    vn_only,
)

@pytest.fixture
def r04b09_diffusion_config(
    ndyn_substeps,  # noqa: F811 # imported `ndyn_substeps` fxiture
) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
    )


@pytest.fixture
def r04b09_iconrun_config(
    ndyn_substeps,  # noqa: F811 # imported `ndyn_substeps` fxiture
) -> IconRunConfig:
    """
    Create IconRunConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return IconRunConfig(
        dtime = 10.0,
        start_date = datetime(2021, 6, 20, 12, 0, 0),
        end_date = datetime(2021, 6, 20, 12, 0, 10),
        is_testcase = False,
        ndyn_substeps = ndyn_substeps,
        apply_horizontal_diff_at_large_dt = True,
        apply_extra_diffusion_for_largeCFL = True,
    )

@pytest.fixture
def timeloop_diffusion_savepoint_init(
    data_provider,  # noqa: F811 # imported fixtures data_provider
    timeloop_date_init,  # noqa: F811 # imported fixtures data_provider
):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'timeloop_date_init'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag is set to true
    """
    return data_provider.from_savepoint_diffusion_init(linit=True, date=timeloop_date_init)


@pytest.fixture
def timeloop_diffusion_savepoint_exit(
    data_provider,  # noqa: F811 # imported fixtures data_provider`
    timeloop_date_exit,  # noqa: F811 # imported fixtures step_date_exit`
):
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'timeloop_date_exit'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=False, date=timeloop_date_exit)
    return sp

@pytest.fixture
def timeloop_velocity_savepoint_init(data_provider, timeloop_date_init, timeloop_istep_init, vn_only, timeloop_jstep_init):  # F811
    """
    Load data from ICON savepoint at start of velocity_advection module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_init(
        istep=timeloop_istep_init, vn_only=vn_only, date=timeloop_date_init, jstep=timeloop_jstep_init
    )


@pytest.fixture
def timeloop_nonhydro_savepoint_init(data_provider, timeloop_date_init, timeloop_istep_init, timeloop_jstep_init):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_init(istep=timeloop_istep_init, date=timeloop_date_init, jstep=timeloop_jstep_init)


@pytest.fixture
def timeloop_nonhydro_savepoint_exit(data_provider, timeloop_date_exit, timeloop_istep_exit, timeloop_jstep_exit):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_exit(istep=timeloop_istep_exit, date=timeloop_date_exit, jstep=timeloop_jstep_exit)


@pytest.fixture
def timeloop_nonhydro_step_savepoint_exit(data_provider, timeloop_date_exit, timeloop_jstep_exit):  # noqa F811
    """
    Load data from ICON savepoint at final exit (after predictor and corrector, and 3 final stencils) of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_step_exit(date=timeloop_date_exit, jstep=timeloop_jstep_exit)

@pytest.fixture
def timeloop_istep_init():
    return 1


@pytest.fixture
def timeloop_jstep_init():
    return 0

@pytest.fixture
def timeloop_istep_exit():
    return 2


@pytest.fixture
def timeloop_jstep_exit():
    return 1
