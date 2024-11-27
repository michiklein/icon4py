# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _cloud_to_snow(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    ns:       fa.CellField[ta.wpfloat],             # Snow number
    lam:      fa.CellField[ta.wpfloat],             # Snow slope parameter (lambda)
    V1S:      ta.wpfloat,
    V0S:      ta.wpfloat,
    TFRZ_HOM: ta.wpfloat,
    QMIN:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                     # Return: Riming snow rate
    ECS = 0.9
    B_RIM = -(V1S + 3.0)
    C_RIM = 2.61 * ECS * V0S            # (with pi*gam(v1s+3)/4 = 2.610)
    return where( (minimum(qc,qs) > QMIN) & (t > TFRZ_HOM), C_RIM*ns*qc*power(lam, B_RIM), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_snow(
    t:                fa.CellField[ta.wpfloat],             # Temperature
    qc:               fa.CellField[ta.wpfloat],             # Cloud specific mass
    qs:               fa.CellField[ta.wpfloat],             # Snow specific mass
    ns:               fa.CellField[ta.wpfloat],             # Snow number
    lam:              fa.CellField[ta.wpfloat],             # Snow slope parameter
    V1S:              ta.wpfloat,
    V0S:              ta.wpfloat,
    TFRZ_HOM:         ta.wpfloat,
    QMIN:             ta.wpfloat,
    riming_snow_rate: fa.CellField[ta.wpfloat],             # output
):
    _cloud_to_snow(t, qc, qs, ns, lam, V1S, V0S, TFRZ_HOM, QMIN, out=riming_snow_rate)
