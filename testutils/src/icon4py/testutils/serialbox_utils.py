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
import serialbox as ser
from functional.common import Dimension
from functional.ffront.fbuiltins import int32
from functional.iterator.embedded import np_as_located_field

from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2VDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)


class IconDiffusionSavepoint:
    def __init__(self, sp: ser.Savepoint, ser: ser.Serializer):
        self.savepoint = sp
        self.serializer = ser

    def print_meta_info(self):
        print(self.savepoint.metainfo)

    def _get_field(self, name, *dimensions, dtype=float):
        buffer = np.squeeze(self.serializer.read(name, self.savepoint).astype(dtype))
        print(f"{name} {buffer.shape}")
        return np_as_located_field(*dimensions)(buffer)

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n: metadata[n] for n in names if n in metadata}


class IconDiffusionInitSavepoint(IconDiffusionSavepoint):
    def vct_a(self):
        return self._get_field("vct_a", KDim)

    def tangent_orientation(self):
        return self._get_field("tangent_orientation", EdgeDim)

    def inverse_primal_edge_lengths(self):
        return self._get_field("inv_primal_edge_length", EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", VertexDim, E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", VertexDim, E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", VertexDim, E2C2VDim)

    def cell_areas(self):
        return self._get_field("cell_areas", CellDim)

    def edge_areas(self):
        return self._get_field("cell_areas", EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", EdgeDim)

    def cells_start_index(self):
        return self._read_int32_shift1("c_start_index")

    def _read_int32_shift1(self, name: str):
        """
        Read a index field and shift it by -1.

        use for start indeces: the shift accounts for the zero based python
        values are converted to int32
        """
        return (self.serializer.read(name, self.savepoint) - 1).astype(int32)

    def cells_end_index(self):
        return self._read_int32("c_end_index")

    def read_int(self, name: str):
        buffer = self.serializer.read(name, self.savepoint).astype(int)
        print(f"{name} {buffer.shape}")
        return buffer

    def _read_int32(self, name: str):
        """
        Read a int field by name.

        use this for end indices: because FORTRAN slices  are inclusive [from:to] _and_ one based
        this accounts for being exclusive python exclusive bounds: [from:to)
        field values are convert to int32
        """
        return self.serializer.read(name, self.savepoint).astype(int32)

    def vertex_start_index(self):
        return self._read_int32_shift1("v_start_index")

    def vertex_end_index(self):
        return self._read_int32("v_end_index")

    def edge_start_index(self):
        return self._read_int32_shift1("e_start_index")

    def edge_end_index(self):
        # don't need to subtract 1, because FORTRAN slices  are inclusive [from:to] so the being
        # one off accounts for being exclusive [from:to)
        return self.serializer.read("e_end_index", self.savepoint)

    def refin_ctrl(self, dim: Dimension):
        if dim == CellDim:
            return self.serializer.read("c_refin_ctl", self.savepoint)
        elif dim == EdgeDim:
            return self.serializer.read("e_refin_ctl", self.savepoint)
        elif dim == VertexDim:
            return self.serializer.read("v_refin_ctl", self.savepoint)
        else:
            return None

    def c2e(self):
        # subtract 1 to account for python being 0 based
        return self.serializer.read("c2e", self.savepoint) - 1

    def c2e2c(self):
        # subtract 1 to account for python being 0 based
        return self.serializer.read("c2e2c", self.savepoint) - 1

    def e2c(self):
        # subtract 1 to account for python being 0 based
        return self.serializer.read("e2c", self.savepoint) - 1

    def e2v(self):
        # subtract 1 to account for python being 0 based
        return self.serializer.read("e2v", self.savepoint) - 1

    def v2e(self):
        # subtract 1 to account for python being 0 based
        return self.serializer.read("v2e", self.savepoint) - 1

    def hdef_ic(self):
        return self._get_field("hdef_ic", CellDim, KDim)

    def div_ic(self):
        return self._get_field("div_ic", CellDim, KDim)

    def dwdx(self):
        return self._get_field("dwdx", CellDim, KDim)

    def dwdy(self):
        return self._get_field("dwdy", CellDim, KDim)

    def vn(self):
        return self._get_field("vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("w", CellDim, KDim)

    def exner(self):
        return self._get_field("exner", CellDim, KDim)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", CellDim, KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", CellDim, KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", EdgeDim, KDim)

    def mask_diff(self):
        return self._get_field("mask_hdiff", CellDim, KDim, dtype=int)

    def zd_diffcoef(self):
        return self._get_field("zd_diffcoef", CellDim, KDim)

    def zd_intcoef(self):
        return self._get_field("vcoef", CellDim, C2E2CDim, KDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", CellDim, C2EDim)

    def geofac_div(self):
        return self._get_field("geofac_div", CellDim, C2EDim)

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", CellDim, C2E2CODim)

    def geofac_grg(self):
        grg = np.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        return np_as_located_field(CellDim, C2E2CODim)(
            grg[:, :, 0]
        ), np_as_located_field(CellDim, C2E2CODim)(grg[:, :, 1])

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", EdgeDim)

    def zd_vertidx(self):
        return self._get_field("zd_vertidx", CellDim, C2E2CDim, KDim, dtype=int)

    def rbf_vec_coeff_v1(self):
        return self._get_field("rbf_vec_coeff_v1", VertexDim, V2EDim)

    def rbf_vec_coeff_v2(self):
        return self._get_field("rbf_vec_coeff_v2", VertexDim, V2EDim)

    def diff_multfac_smag(self):
        return np.squeeze(self.serializer.read("diff_multfac_smag", self.savepoint))

    def smag_limit(self):
        return np.squeeze(self.serializer.read("smag_limit", self.savepoint))

    def diff_multfac_n2w(self):
        return np.squeeze(self.serializer.read("diff_multfac_n2w", self.savepoint))

    def nrdmax(self):
        return self.serializer.read("nrdmax", self.savepoint)[0]

    def nudgezone_diff(self) -> int:
        return self.serializer.read("nudgezone_diff", self.savepoint)[0]

    def bdy_diff(self) -> int:
        return self.serializer.read("bdy_diff", self.savepoint)[0]

    def fac_bdydiff_v(self) -> int:
        return self.serializer.read("fac_bdydiff_v", self.savepoint)[0]

    def smag_offset(self):
        return self.serializer.read("smag_offset", self.savepoint)[0]

    def diff_multfac_w(self):
        return self.serializer.read("diff_multfac_w", self.savepoint)[0]

    def diff_multfac_vn(self):
        return self.serializer.read("diff_multfac_vn", self.savepoint)


class IconDiffusionExitSavepoint(IconDiffusionSavepoint):
    def vn(self):
        return self._get_field("x_vn", EdgeDim, KDim)

    def theta_v(self):
        return self._get_field("x_theta_v", CellDim, KDim)

    def w(self):
        return self._get_field("x_w", CellDim, KDim)

    def exner(self):
        return self._get_field("x_exner", CellDim, KDim)


class IconSerialDataProvider:
    def __init__(self, fname_prefix, path=".", do_print=False):
        self.rank = 0
        self.serializer: ser.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{str(self.rank)}"
        self._init_serializer(do_print)

    def _init_serializer(self, do_print: bool):
        if not self.fname:
            print(" WARNING: no filename! closing serializer")
        self.serializer = ser.Serializer(
            ser.OpenModeKind.Read, self.file_path, self.fname
        )
        if do_print:
            self.print_info()

    def print_info(self):
        print(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        print(f"FIELDNAMES: {self.serializer.fieldnames()}")

    def from_savepoint_init(self, linit: bool, date: str) -> IconDiffusionInitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-init"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconDiffusionInitSavepoint(savepoint, self.serializer)

    def from_save_point_exit(
        self, linit: bool, date: str
    ) -> IconDiffusionExitSavepoint:
        savepoint = (
            self.serializer.savepoint["call-diffusion-exit"]
            .linit[linit]
            .date[date]
            .as_savepoint()
        )
        return IconDiffusionExitSavepoint(savepoint, self.serializer)
