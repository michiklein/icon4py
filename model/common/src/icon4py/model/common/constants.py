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
import sys
from typing import Final


#: Gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90),
#: see https://glossary.ametsoc.org/wiki/Gas_constant.
GAS_CONSTANT_DRY_AIR: Final[float] = 287.04
RD: Final[float] = GAS_CONSTANT_DRY_AIR

#: Specific heat at constant pressure [J/K/kg]
CPD: Final[float] = 1004.64

#: [J/K/kg] specific heat at constant volume
CVD: Final[float] = CPD - RD
CVD_O_RD: Final[float] = CVD / RD

#: Gas constant for water vapor [J/K/kg], rv in ICON.
GAS_CONSTANT_WATER_VAPOR: Final[float] = 461.51
RV: Final[float] = GAS_CONSTANT_WATER_VAPOR

#: Av. gravitational acceleration [m/s^2]
GRAVITATIONAL_ACCELERATION: Final[float] = 9.80665
GRAV: Final[float] = GRAVITATIONAL_ACCELERATION

#: reference pressure for Exner function [Pa]
P0REF: Final[float] = 100000.0


# Math constants
dbl_eps = sys.float_info.epsilon  # EPSILON(1._wp)

# Implementation constants
#: default physics to dynamics time step ratio
# TODO (magdalena) not a constant, this is a default config parameter
DEFAULT_PHYSICS_DYNAMICS_TIMESTEP_RATIO: Final[float] = 5.0

#: Klemp (2008) type Rayleigh damping
# TODO (magdalena) not a constant, move somewhere else, convert to enum
RAYLEIGH_KLEMP: Final[int] = 2

# Constants for physics schemes
#: Melting temperature of ice/snow [K]
TMELT: Final[float] = 273.15

#: Latent heat of vaporisation for water [J/kg]
ALV: Final[float] = 2.5008e6

#: Latent heat of sublimation for water [J/kg]
ALS: Final[float] = 2.8345e6

#: latent heat of fusion for water [J/kg]
ALF: Final[float] = ALS - ALV

#: cp_d / cp_l - 1
RCPL: Final[float] = 3.1733

CLW: Final[float] = (RCPL + 1.0) * CPD

#: Inverse of specific heat at constant pressure  [K*kg/J]
RCPD = 1.0 / CPD

#: Inverse of specific heat at constant volume [K*kg/J]
RCVD = 1.0 / CVD


