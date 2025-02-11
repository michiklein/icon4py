# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, minimum, maximum, power, exp, sqrt
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.frozen import idx, g_ct, t_d

@gtx.field_operator
def _cloud_to_graupel(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:      fa.CellKField[ta.wpfloat],             # Ambient density
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qg:       fa.CellKField[ta.wpfloat],             # Graupel specific mass
) -> fa.CellKField[ta.wpfloat]:                     # Return: Riming graupel rate
    A_RIM = 4.43
    B_RIM = 0.94878
    return where( (minimum(qc,qg) > g_ct.qmin) & (t > g_ct.tfrz_hom), A_RIM * qc * power(qg * rho, B_RIM), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_graupel(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:      fa.CellKField[ta.wpfloat],             # Ambient density
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qg:       fa.CellKField[ta.wpfloat],             # Graupel specific mass
    riming_graupel_rate:     fa.CellKField[ta.wpfloat],             # output
):
    _cloud_to_graupel(t, rho, qc, qg, out=riming_graupel_rate)

@gtx.field_operator
def _cloud_to_rain(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qr:       fa.CellKField[ta.wpfloat],             # Rain water specific mass
    nc:       ta.wpfloat,                            # Cloud water number concentration
) -> fa.CellKField[ta.wpfloat]:                      # Return: Riming graupel rate
    QMIN_AC       = 1.0e-6                           # threshold for auto conversion
    TAU_MAX       = 0.90e0                           # maximum allowed value of tau
    TAU_MIN       = 1.0e-30                          # minimum allowed value of tau
    A_PHI         = 6.0e2                            # constant in phi-function for autoconversion
    B_PHI         = 0.68e0                           # exponent in phi-function for autoconversion
    C_PHI         = 5.0e-5                           # exponent in phi-function for accretion
    AC_KERNEL     = 5.25e0                           # kernel coeff for SB2001 accretion
    X3            = 2.0e0                            # gamma exponent for cloud distribution
    X2            = 2.6e-10                          # separating mass between cloud and rain
    X1            = 9.44e9                           # kernel coeff for SB2001 autoconversion
    AU_KERNEL     = X1/(20.0*X2) * (X3+2.0) * (X3+4.0) / ((X3+1.0)*(X3+1.0))

    # TO-DO: put as much of this into the WHERE statement as possible
    tau = maximum(TAU_MIN, minimum(1.0-qc/(qc+qr), TAU_MAX))  # temporary cannot go in where
    phi = power(tau,B_PHI)
    phi = A_PHI * phi * power(1.0-phi, 3.0)
    xau = AU_KERNEL * power(qc*qc/nc, 2.0) * (1.0 + phi/power(1.0-tau,2.0))
    xac = AC_KERNEL * qc * qr * power(tau/(tau+C_PHI),4.0)
    return where( (qc > QMIN_AC) & (t > g_ct.tfrz_hom), xau+xac , 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_rain(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass 
    qr:       fa.CellKField[ta.wpfloat],             # Rain water specific mass
    nc:       ta.wpfloat,                            # Cloud water number concentration
    conversion_rate:  fa.CellKField[ta.wpfloat],     # output
):
    _cloud_to_rain(t, qc, qr, nc, out=conversion_rate)

@gtx.field_operator
def _cloud_to_snow(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
    ns:       fa.CellKField[ta.wpfloat],             # Snow number
    lam:      fa.CellKField[ta.wpfloat],             # Snow slope parameter (lambda)
) -> fa.CellKField[ta.wpfloat]:                     # Return: Riming snow rate
    ECS = 0.9
    B_RIM = -(g_ct.v1s + 3.0)
    C_RIM = 2.61 * ECS * g_ct.v0s            # (with pi*gam(v1s+3)/4 = 2.610)
    return where( (minimum(qc,qs) > g_ct.qmin) & (t > g_ct.tfrz_hom), C_RIM*ns*qc*power(lam, B_RIM), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_snow(
    t:                fa.CellKField[ta.wpfloat],             # Temperature
    qc:               fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qs:               fa.CellKField[ta.wpfloat],             # Snow specific mass
    ns:               fa.CellKField[ta.wpfloat],             # Snow number
    lam:              fa.CellKField[ta.wpfloat],             # Snow slope parameter
    riming_snow_rate: fa.CellKField[ta.wpfloat],             # output
):
    _cloud_to_snow(t, qc, qs, ns, lam, out=riming_snow_rate)

@gtx.field_operator
def _cloud_x_ice(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    qc:        fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qi:        fa.CellKField[ta.wpfloat],             # Ice specific mass
    dt:        ta.wpfloat,                           # time step
) -> fa.CellKField[ta.wpfloat]:                       # Homogeneous freezing rate

    result = where( (qc > g_ct.qmin) & (t < g_ct.tfrz_hom), qc / dt, 0. )
    result = where( (qi > g_ct.qmin) & (t > t_d.tmelt), -qi/dt, result )
    return result

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_x_ice(
    t:                 fa.CellKField[ta.wpfloat],     # Temperature
    qc:                fa.CellKField[ta.wpfloat],     # Cloud specific mass
    qi:                fa.CellKField[ta.wpfloat],     # Ice specific mass
    dt:                ta.wpfloat,                   # time step
    freezing_rate:     fa.CellKField[ta.wpfloat],     # output
):
    _cloud_x_ice(t, qc, qi, dt, out=freezing_rate)

@gtx.field_operator
def _graupel_to_rain(
    t:       fa.CellKField[ta.wpfloat],             # Ambient temperature
    p:       fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:     fa.CellKField[ta.wpfloat],             # Ambient density
    dvsw0:   fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    qg:      fa.CellKField[ta.wpfloat],             # Graupel specific mass
) -> fa.CellKField[ta.wpfloat]:                     # Return: rain rate
    A_MELT  = g_ct.tx - 389.5                           # melting prefactor
    B_MELT  = 0.6                                  # melting exponent
    C1_MELT = 12.31698                             # Constants in melting formula
    C2_MELT = 7.39441e-05                          # Constants in melting formula
    return where( (t > maximum(t_d.tmelt,t_d.tmelt-g_ct.tx*dvsw0)) & (qg > g_ct.qmin), (C1_MELT/p + C2_MELT)*(t-t_d.tmelt+A_MELT*dvsw0)*power(qg*rho,B_MELT), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_to_rain(
    t:       fa.CellKField[ta.wpfloat],             # Ambient temperature
    p:       fa.CellKField[ta.wpfloat],             # Ambient pressue
    rho:     fa.CellKField[ta.wpfloat],             # Ambient density
    dvsw0:   fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    qg:      fa.CellKField[ta.wpfloat],             # Graupel specific mass
    rain_rate:   fa.CellKField[ta.wpfloat],             # output
):
    _graupel_to_rain(t, p, rho, dvsw0, qg, out=rain_rate)

@gtx.field_operator
def _ice_to_graupel(
    rho:          fa.CellKField[ta.wpfloat],             # Ambient density
    qr:           fa.CellKField[ta.wpfloat],             # Rain specific mass
    qg:           fa.CellKField[ta.wpfloat],             # Graupel specific mass
    qi:           fa.CellKField[ta.wpfloat],             # Ice specific mass
    sticking_eff: fa.CellKField[ta.wpfloat],             # Sticking efficiency
) -> fa.CellKField[ta.wpfloat]:                          # Aggregation of ice by graupel
    A_CT     = 1.72                                     # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)    
    B_CT     = 0.875                                    # Exponent = 7/8
    C_AGG_CT = 2.46
    B_AGG_CT = 0.94878                                  # Exponent 
    result = where( (qi > g_ct.qmin) & (qg > g_ct.qmin), sticking_eff * qi * C_AGG_CT * power(rho*qg, B_AGG_CT), 0. )
    result = where( (qi > g_ct.qmin) & (qr > g_ct.qmin), result + A_CT*qi*power(rho*qr, B_CT), result )
    return result

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_to_graupel(
    rho:          fa.CellKField[ta.wpfloat],             # Ambient density
    qr:           fa.CellKField[ta.wpfloat],             # Rain specific mass
    qg:           fa.CellKField[ta.wpfloat],             # Graupel specific mass
    qi:           fa.CellKField[ta.wpfloat],             # Ice specific mass
    sticking_eff: fa.CellKField[ta.wpfloat],             # Sticking efficiency
    aggregation:  fa.CellKField[ta.wpfloat],             # output
):
    _ice_to_graupel(rho, qr, qg, qi, sticking_eff, out=aggregation)

@gtx.field_operator
def _ice_to_snow(
    qi:           fa.CellKField[ta.wpfloat],             # Ice specific mass
    ns:           fa.CellKField[ta.wpfloat],             # Snow number
    lam:          fa.CellKField[ta.wpfloat],             # Snow intercept parameter
    sticking_eff: fa.CellKField[ta.wpfloat],             # Sticking efficiency
) -> fa.CellKField[ta.wpfloat]:                          # Conversion rate of ice to snow
    QI0      = 0.0                                      # Critical ice required for autoconversion
    C_IAU    = 1.0e-3                                   # Coefficient of auto conversion
    C_AGG    = 2.61*g_ct.v0s                                 # Coeff of aggregation (2.610 = pi*gam(v1s+3)/4)
    B_AGG    = -(g_ct.v1s + 3.0)                             # Aggregation exponent
    
    return where( (qi > g_ct.qmin), sticking_eff * (C_IAU * maximum(0.0, (qi-QI0)) + qi * (C_AGG * ns) * power(lam, B_AGG)), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_to_snow(
    qi:              fa.CellKField[ta.wpfloat],             # Ice specific mass
    ns:              fa.CellKField[ta.wpfloat],             # Snow number
    lam:             fa.CellKField[ta.wpfloat],             # Snow intercept parameter
    sticking_eff:    fa.CellKField[ta.wpfloat],             # Sticking efficiency
    conversion_rate: fa.CellKField[ta.wpfloat],             # output
):
    _ice_to_snow(qi, ns, lam, sticking_eff,  out=conversion_rate)

@gtx.field_operator
def _rain_to_graupel(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qc:        fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qr:        fa.CellKField[ta.wpfloat],             # Specific humidity of rain
    qi:        fa.CellKField[ta.wpfloat],             # Ice specific mass
    qs:        fa.CellKField[ta.wpfloat],             # Snow specific mass
    mi:        fa.CellKField[ta.wpfloat],             # Ice crystal mass
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water (T)
    dt:        ta.wpfloat,                           # time step
) -> fa.CellKField[ta.wpfloat]:                       # Conversion rate from graupel to rain

    TFRZ_RAIN = t_d.tmelt - 2.0
    A1        = 9.95e-5            # coefficient for immersion raindrop freezing: alpha_if
    B1        = 1.75               # coefficient for immersion raindrop freezing: a_if
    C1        = 1.68               # coefficient for raindrop freezing
    C2        = 0.66               # coefficient for immersion raindrop freezing: a_if
    C3        = 1.0                # coefficient for immersion raindrop freezing: a_if
    C4        = 0.1                # coefficient for immersion raindrop freezing: a_if
    A2        = 1.24e-3            # (PI/24)*EIR*V0R*Gamma(6.5)*AR**(-5/8)
    B2        = 1.625              # TBD
    QS_CRIT   = 1.0e-7             # TBD

    mask   = (qr > g_ct.qmin) & (t < TFRZ_RAIN)
    result = where( mask & (dvsw+qc <= 0.0), (exp(C2*(TFRZ_RAIN-t))-C3) * (A1 * power((qr * rho), B1)), 0. )
    result = where( mask & (t <= g_ct.tfrz_hom), qr/dt, 0. )

    return where( (minimum(qi,qr) > g_ct.qmin) & (qs > QS_CRIT), result + A2*(qi/mi)*power((rho*qr), B2), result)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rain_to_graupel(
    t:               fa.CellKField[ta.wpfloat],       # Temperature
    rho:             fa.CellKField[ta.wpfloat],       # Ambient density
    qc:              fa.CellKField[ta.wpfloat],       # Cloud specific mass
    qr:              fa.CellKField[ta.wpfloat],       # Specific humidity of rain
    qi:              fa.CellKField[ta.wpfloat],       # Ice specific mass
    qs:              fa.CellKField[ta.wpfloat],       # Snow specific mass
    mi:              fa.CellKField[ta.wpfloat],       # Ice crystal mass
    dvsw:            fa.CellKField[ta.wpfloat],       # qv-qsat_water (T)
    dt:              ta.wpfloat,                     # time step
    conversion_rate: fa.CellKField[ta.wpfloat],       # output
):
    _rain_to_graupel(t, rho, qc, qr, qi, qs, mi, dvsw, dt, out=conversion_rate)

@gtx.field_operator
def _rain_to_vapor(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qc:        fa.CellKField[ta.wpfloat],             # Cloud-specific humidity
    qr:        fa.CellKField[ta.wpfloat],             # Rain-specific humidity 
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water (T)
    dt:        ta.wpfloat,                           # time step
) -> fa.CellKField[ta.wpfloat]:                       # Conversion rate from graupel to rain

    B1_RV     =  0.16667           # exponent in power-law relation for mass density
    B2_RV     =  0.55555           # TBD
    C1_RV     =  0.61              # TBD
    C2_RV     = -0.0163            # TBD
    C3_RV     =  1.111e-4          # TBD
    A1_RV     =  1.536e-3          # TBD
    A2_RV     =  1.0e0             # TBD
    A3_RV     = 19.0621e0          # TBD

    # TO-DO: move as much as possible into WHERE statement
    tc = t - t_d.tmelt
    evap_max = (C1_RV + tc * (C2_RV + C3_RV*tc)) * (-dvsw) / dt 
    return where( (qr > g_ct.qmin) & (dvsw+qc <= 0.0), minimum(A1_RV * (A2_RV+A3_RV*power(qr*rho,B1_RV)) * (-dvsw) * power(qr*rho,B2_RV), evap_max), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def rain_to_vapor(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qc:        fa.CellKField[ta.wpfloat],             # Cloud-specific humidity
    qr:        fa.CellKField[ta.wpfloat],             # Rain-specific humidity
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water (T)
    dt:        ta.wpfloat,                           # time step
    conversion_rate: fa.CellKField[ta.wpfloat],       # output
):
    _rain_to_vapor(t, rho, qc, qr, dvsw, dt, out=conversion_rate)

@gtx.field_operator
def _snow_to_graupel(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:      fa.CellKField[ta.wpfloat],             # Ambient density
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
) -> fa.CellKField[ta.wpfloat]:                      # Return: Riming snow rate
    A_RIM_CT = 0.5                                  # Constants in riming formula
    B_RIM_CT = 0.75
    return where( (minimum(qc,qs) > g_ct.qmin) & (t > g_ct.tfrz_hom), A_RIM_CT * qc * power(qs*rho, B_RIM_CT), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_to_graupel(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:      fa.CellKField[ta.wpfloat],             # Ambient density
    qc:       fa.CellKField[ta.wpfloat],             # Cloud specific mass
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
    conversion_rate: fa.CellKField[ta.wpfloat],      # output
):
    _snow_to_graupel(t, rho, qc, qs, out=conversion_rate)

@gtx.field_operator
def _snow_to_rain(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    p:        fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:      fa.CellKField[ta.wpfloat],             # Ambient density
    dvsw0:    fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
) -> fa.CellKField[ta.wpfloat]:                      # Return: Riming snow rate
    C1_SR = 79.6863                                 # Constants in melting formula
    C2_SR = 0.612654e-3                             # Constants in melting formula
    A_SR  = g_ct.tx - 389.5                              # Melting prefactor
    B_SR  = 0.8                                     # Melting exponent
    return where( (t > maximum(t_d.tmelt, t_d.tmelt-g_ct.tx*dvsw0)) & (qs > g_ct.qmin), (C1_SR/p + C2_SR) * (t - t_d.tmelt + A_SR*dvsw0) * power(qs*rho, B_SR), 0.0)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_to_rain(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    p:        fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:      fa.CellKField[ta.wpfloat],             # Ambient density
    dvsw0:    fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
    conversion_rate: fa.CellKField[ta.wpfloat],      # output
):
    _snow_to_rain(t, p, rho, dvsw0, qs, out=conversion_rate)

@gtx.field_operator
def _vapor_x_graupel(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    p:         fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qg:        fa.CellKField[ta.wpfloat],             # Graupel specific mass
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellKField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
) -> fa.CellKField[ta.wpfloat]:                       # Homogeneous freezing rate
    A1_VG = 0.398561
    A2_VG = -0.00152398
    A3    = 2554.99
    A4    = 2.6531E-7
    A5    = 0.153907
    A6    = -7.86703e-07
    A7    = 0.0418521
    A8    = -4.7524E-8
    B_VG  = 0.6
    result = where( (qg > g_ct.qmin) & (t < t_d.tmelt), (A1_VG + A2_VG*t + A3/p + A4*p) * dvsi * power(qg*rho, B_VG), 0. )
    result = where( (qg > g_ct.qmin) & (t >= t_d.tmelt) & (t > (t_d.tmelt - g_ct.tx*dvsw0)), (A5 + A6*p) * minimum(0.0, dvsw0) * power(qg*rho, B_VG), result )
    result = where( (qg > g_ct.qmin) & (t >= t_d.tmelt) & (t <= (t_d.tmelt - g_ct.tx*dvsw0)), (A7 + A8*p) * dvsw * power(qg*rho, B_VG), result )
    return maximum(result, -qg/dt)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_graupel(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    p:         fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qg:        fa.CellKField[ta.wpfloat],             # Graupel specific mass
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellKField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
    exchange_rate:     fa.CellKField[ta.wpfloat],     # output
):
    _vapor_x_graupel(t, p, rho, qg, dvsw, dvsi, dvsw0, dt, out=exchange_rate)

@gtx.field_operator
def _vapor_x_ice(
    qi:        fa.CellKField[ta.wpfloat],             # Specific humidity of ice
    mi:        fa.CellKField[ta.wpfloat],             # Ice crystal mass
    eta:       fa.CellKField[ta.wpfloat],             # Deposition factor
    dvsi:      fa.CellKField[ta.wpfloat],             # Vapor excess qv-qsat_ice(T)
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    dt:        ta.wpfloat,                           # time step
) -> fa.CellKField[ta.wpfloat]:                       # Rate of vapor deposition to ice
    AMI    = 130.0                 # Formfactor for mass-size relation of cold ice
    B_EXP  = -0.67                 # exp. for conv. (-1 + 0.33) of ice mass to sfc area
    A_FACT = 4.0 * AMI**(-1.0/3.0)  

    # TO-DO: see if this can be folded into the WHERE statement
    mask   = (A_FACT * eta) * rho * qi * power(mi, B_EXP) * dvsi
    return where( mask > 0.0, minimum(mask, dvsi/dt), maximum(maximum(mask, dvsi/dt), -qi/dt) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_ice(
    qi:        fa.CellKField[ta.wpfloat],             # Specific humidity of ice
    mi:        fa.CellKField[ta.wpfloat],             # Ice crystal mass
    eta:       fa.CellKField[ta.wpfloat],             # Deposition factor
    dvsi:      fa.CellKField[ta.wpfloat],             # Vapor excess qv-qsat_ice(T)
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    dt:        ta.wpfloat,                           # time step
    vapor_deposition_rate: fa.CellKField[ta.wpfloat]  # output
):
    _vapor_x_ice(qi, mi, eta, dvsi, rho, dt, out=vapor_deposition_rate)

@gtx.field_operator
def _vapor_x_snow(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    p:         fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qs:        fa.CellKField[ta.wpfloat],             # Snow specific mass
    ns:        fa.CellKField[ta.wpfloat],             # Snow number
    lam:       fa.CellKField[ta.wpfloat],             # Slope parameter (lambda) snow
    eta:       fa.CellKField[ta.wpfloat],             # Deposition factor
    ice_dep:   fa.CellKField[ta.wpfloat],             # Limiter for vapor dep on snow
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellKField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
) -> fa.CellKField[ta.wpfloat]:                       # Rate of vapor deposition to snow
    NU     = 1.75e-5;                                # kinematic viscosity of air
    A0_VS  = 1.0
    A1_VS  = 0.4182 * sqrt(g_ct.v0s/NU)
    A2_VS  = -(g_ct.v1s + 1.0) / 2.0
    EPS    = 1.0e-15
    QS_LIM = 1.0e-7
    CNX    = 4.0
    B_VS   = 0.8
    C1_VS  = 31282.3
    C2_VS  = 0.241897
    C3_VS  = 0.28003
    C4_VS  = -0.146293E-6

    # See if this can be incorporated into WHERE statement
    mask = (CNX * ns * eta / rho) * (A0_VS + A1_VS * power(lam, A2_VS)) * dvsi / (lam * lam + EPS)

    # GZ: This mask>0 limitation, which was missing in the original graupel scheme,
    # is crucial for numerical stability in the tropics!
    # a meaningful distinction between cloud ice and snow
    result = where( (qs > g_ct.qmin) & (t < t_d.tmelt) & (mask > 0.0), minimum(mask, dvsi/dt - ice_dep), 0.0 ) 
    result = where( (qs > g_ct.qmin) & (t < t_d.tmelt) & (qs <= QS_LIM), minimum(result, 0.0), result )
    # ELSE section
    result = where( (qs > g_ct.qmin) & (t >= t_d.tmelt) & (t > (t_d.tmelt - g_ct.tx*dvsw0)), (C1_VS/p + C2_VS) * minimum(0.0, dvsw0) * power(qs*rho, B_VS), 0.0)
    result = where( (qs > g_ct.qmin) & (t >= t_d.tmelt) & (t <= (t_d.tmelt - g_ct.tx*dvsw0)), (C3_VS + C4_VS*p) * dvsw * power(qs*rho, B_VS), 0.0)
    return where( (qs > g_ct.qmin), maximum(result, -qs/dt), 0.0)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vapor_x_snow(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    p:         fa.CellKField[ta.wpfloat],             # Ambient pressure
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    qs:        fa.CellKField[ta.wpfloat],             # Snow specific mass
    ns:        fa.CellKField[ta.wpfloat],             # Snow number
    lam:       fa.CellKField[ta.wpfloat],             # Slope parameter (lambda) snow
    eta:       fa.CellKField[ta.wpfloat],             # Deposition factor
    ice_dep:   fa.CellKField[ta.wpfloat],             # Limiter for vapor dep on snow
    dvsw:      fa.CellKField[ta.wpfloat],             # qv-qsat_water(T)
    dvsi:      fa.CellKField[ta.wpfloat],             # qv-qsat_ice(T)
    dvsw0:     fa.CellKField[ta.wpfloat],             # qv-qsat_water(T0)
    dt:        ta.wpfloat,                           # time step
    vapor_deposition_rate:     fa.CellKField[ta.wpfloat],     # output
):
    _vapor_x_snow(t, p, rho, qs, ns, lam, eta, ice_dep, dvsw, dvsi, dvsw0, dt, out=vapor_deposition_rate)
