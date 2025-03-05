import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.diffusion.stencils.calculate_e2c2v_simple import (
    c2v_stecil,
    e2c_stencil,
    e2c2v_stencil,
)

def _c2v_stencil_numpy(vertex_input: np.ndarray) -> np.ndarray:
    return np.sum(vertex_input, axis=0)

def _e2c_stencil_numpy(cell_input: np.ndarray) -> np.ndarray:
    return np.sum(cell_input, axis=0)

def _e2c2v_stencil_numpy(vertex_input: np.ndarray) -> np.ndarray:
    tmp = _c2v_stencil_numpy(vertex_input)
    return _e2c_stencil_numpy(tmp)

class TestCalculateE2C2V:
    