from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import numpy as np
import time
import netCDF4


def get_coords_v(grid):
    nc = netCDF4.Dataset(grid, mode="r")
    
    
def get_coords_e(grid):
    
def get_coords_c(grid):


def trim_grid_file(grid):
    
    
def reorder_neighbor_tables(grid):

def get_torus_cartesian_dimensions(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    sorted_y_coordinates = np.sort(nc["cartesian_y_vertices"][:])
    longitude_dimension = np.count_nonzero(sorted_y_coordinates == 0.0)
    latitude_dimension = int(len(sorted_y_coordinates) / longitude_dimension)
    return (longitude_dimension, latitude_dimension)

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



grid_file = "/Users/michaelklein/Documents/MASTERTHESIS/all_torus_files/big_torus_100000_100000_64.nc"

# Load the torus grid
torus_grid_raw = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())
torus_grid_trimmed = trim_grid_file(torus_grid_raw)
torus_grid = reorder_neighbor_tables(torus_grid_trimmed)



