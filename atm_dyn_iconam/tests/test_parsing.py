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

from icon4py.diffusion.wrapper.parsing import parse_functions_from_module


def test_parse_functions():
    path = "icon4py.diffusion.wrapper.diffusion_wrapper"
    plugin = parse_functions_from_module(path, ["diffusion_init", "diffusion_run"])
    assert plugin.name == "diffusion_wrapper"
    assert len(plugin.functions) == 2
    assert "diffusion_init" in map(lambda f: f.name, plugin.functions)
    assert "diffusion_run" in map(lambda f: f.name, plugin.functions)
