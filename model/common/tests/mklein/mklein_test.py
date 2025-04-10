from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import numpy as np
import time
import netCDF4


def get_torus_cartesian_dimensions(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    sorted_y = np.sort(nc["cartesian_y_vertices"][:])
    dim_x = np.count_nonzero(sorted_y == 0.0)
    dim_y = int(len(sorted_y) / dim_x)
    nc.close()
    return (dim_x, dim_y)

def get_coords_v(grid):
    nc = netCDF4.Dataset(grid, mode="r")
    x = nc["cartesian_x_vertices"][:]
    y = nc["cartesian_y_vertices"][:]
    nc.close()
    return np.stack((x, y), axis=-1)

def get_coords_e(grid):
    nc = netCDF4.Dataset(grid, mode="r")
    x = nc["edge_middle_cartesian_x"][:]
    y = nc["edge_middle_cartesian_y"][:]
    nc.close()
    return np.stack((x, y), axis=-1)

def get_coords_c(grid):
    nc = netCDF4.Dataset(grid, mode="r")
    x = nc["cell_circumcenter_cartesian_x"][:]
    y = nc["cell_circumcenter_cartesian_y"][:]
    nc.close()
    return np.stack((x, y), axis=-1)

def trim_grid(grid):
    grid_shape = get_torus_cartesian_dimensions(grid)
    v = get_coords_v(grid)
    e = get_coords_e(grid)
    c = get_coords_c(grid)

    nx, ny = grid_shape
    halo = 2

    # make sure halo fits
    usable_x = nx - 2 * halo
    usable_y = ny - 2 * halo
    base = min(usable_x, usable_y)

    # to satisfy christophs idea later
    side = (base // 32) * 32 if base >= 32 else (base // 4) * 4

    # boundary for square
    start_x = (nx - side) // 2
    start_y = (ny - side) // 2
    end_x = start_x + side
    end_y = start_y + side

    # v in square?
    v_idx = np.where(
        (v[:, 0] >= start_x) & (v[:, 0] < end_x) &
        (v[:, 1] >= start_y) & (v[:, 1] < end_y)
    )[0]

    # e in square?
    e_idx = np.where(
        (e[:, 0] >= start_x) & (e[:, 0] < end_x) &
        (e[:, 1] >= start_y) & (e[:, 1] < end_y)
    )[0]

    # c in square?
    c_idx = np.where(
        (c[:, 0] >= start_x) & (c[:, 0] < end_x) &
        (c[:, 1] >= start_y) & (c[:, 1] < end_y)
    )[0]

    return v_idx, e_idx, c_idx, (start_x, end_x, start_y, end_y)

    
    
def reorder_neighbor_tables(grid):

def init_grid_manager(
    fname, num_levels=1, transformation=ToZeroBasedIndexTransformation()
):
    grid_manager = GridManager(
        transformation,
        fname,
        VerticalGridConfig(num_levels),
    )
    grid_manager(None)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    simple_grid = grid_manager.grid
    return simple_grid

def neighbor_sums(grid):
    return None


grid_file = "/Users/michaelklein/Documents/MASTERTHESIS/all_torus_files/big_torus_100000_100000_64.nc"
grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())
vertices, edges, cells, bounds = trim_grid(grid)