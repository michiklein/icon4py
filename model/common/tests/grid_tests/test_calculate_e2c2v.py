import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc

from icon4py.model.common.grid.calculate_e2c2v_simple import (
    c2v_stecil,
    e2c_stencil,
    e2c2v_stencil,
)

from icon4py.model.common.grid.calculate_e2c2v_manualreduction import (
    e2c2v_manualreduction_stencil,
)


def _c2v_stencil_numpy(vertex_input: np.ndarray) -> np.ndarray:
    return np.sum(vertex_input, axis=0)


def _e2c_stencil_numpy(cell_input: np.ndarray) -> np.ndarray:
    return np.sum(cell_input, axis=0)


def _e2c2v_stencil_numpy(vertex_input: np.ndarray) -> np.ndarray:
    tmp = _c2v_stencil_numpy(vertex_input)
    return _e2c_stencil_numpy(tmp)


def _e2c2v_manualreduction_stencil_numpy(vertex_input: np.ndarray) -> np.ndarray:
    return np.array([])


class TestCalculateE2C2V:
    PROGRAMS = {
        "simple": e2c2v_stencil,
        "manualreduction": e2c2v_manualreduction_stencil,
    }
    OUTPUTS = ("edge_out",)
    MARKERS = (pytest.mark.skip_value_error,)

    @staticmethod
    def reference(vertex_input: np.ndarray) -> np.ndarray:
        return _e2c2v_stencil_numpy(vertex_input)

    @pytest.fixture
