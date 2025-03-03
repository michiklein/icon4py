import netCDF4  # type: ignore [import-not-found]
import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]

def init_grid_manager(
    fname,
    num_levels=65,
    transformation=ToZeroBasedIndexTransformation(),
):
    grid_manager = GridManager(
        transformation,
        fname,
        VerticalGridConfig(num_levels),
    )
    grid_manager(None)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(
        filename, num_levels, transformation
    )
    simple_grid = grid_manager.grid
    return simple_grid

def get_vertices_coordinates(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    x_coords = nc["cartesian_x_vertices"][:]
    y_coords = nc["cartesian_y_vertices"][:]
    z_coords = nc["cartesian_z_vertices"][:]
    return np.dstack((x_coords, y_coords, z_coords))[0]


def print_lat_lon_dimensions(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    print("latitude dim: {}".format(len(nc["latitude_vertices"][:])))
    print("longitude dim: {}".format(len(nc["longitude_vertices"][:])))


def print_torus_file_information(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    print(nc)


def get_torus_cartesian_dimensions(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    sorted_y_coordinates = np.sort(nc["cartesian_y_vertices"][:])
    longitude_dimension = np.count_nonzero(sorted_y_coordinates == 0.0)
    print("longitude dim: {}".format(longitude_dimension))
    latitude_dimension = int(len(sorted_y_coordinates) / longitude_dimension)
    print("latitude dim: {}".format(latitude_dimension))
    return (longitude_dimension, latitude_dimension)