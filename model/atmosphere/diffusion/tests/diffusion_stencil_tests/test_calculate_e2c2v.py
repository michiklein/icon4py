import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.diffusion.stencils.calculate_e2c2v_simple import (
    c2v_stecil,
    e2c_stencil,
    e2c2v_stencil,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.testing.helpers import StencilTest


def c2v_numpy(vertex_input: np.ndarray, connectivities: dict) -> np.ndarray:
    c2v_conn = connectivities[dims.C2VDim]
    return np.sum(vertex_input[c2v_conn], axis=1)


def e2c_numpy(cell_input: np.ndarray, connectivities: dict) -> np.ndarray:
    e2c_conn = connectivities[dims.E2CDim]
    return np.sum(cell_input[e2c_conn], axis=1)


def e2c2v_numpy(vertex_input: np.ndarray, connectivities: dict) -> np.ndarray:
    cell_temp = c2v_numpy(vertex_input, connectivities)
    return e2c_numpy(cell_temp, connectivities)


class TestMkleinFirst(StencilTest):
    