import argparse
from os import path
import numpy as np

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    ToZeroBasedIndexTransformation,
)

from plotting import plot_torus  # type: ignore [import-not-found]
from utilities import (
    get_torus_grid,
    get_vertices_coordinates,
    print_torus_file_information,
    print_lat_lon_dimensions,
)  # type: ignore [import-not-found]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    return parser.parse_args()


args = parse_arguments()
grid_file = args.grid

grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())

vertice_coords = get_vertices_coordinates(grid_file)

print_lat_lon_dimensions(grid_file)

print_torus_file_information(grid_file)


def sort_e2v(e2v_table, per_orientation=False):
    out_of_order_edges = []
    for i in range(2, len(e2v_table), 3):
        out_of_order_edges.append(e2v_table[i])
    out_of_order_edges.sort(key=lambda x: x[0])
    sorted_e2v_table = []
    for i in range(len(e2v_table)):
        if i % 3 != 2:
            sorted_e2v_table.append(e2v_table[i].tolist())
        else:
            sorted_e2v_table.append(out_of_order_edges[i // 3].tolist())
    if per_orientation:
        new_sorted_e2v_table = []
        for i in range(0, len(sorted_e2v_table), 3):
            new_sorted_e2v_table.append(sorted_e2v_table[i])
        for i in range(1, len(sorted_e2v_table), 3):
            new_sorted_e2v_table.append(sorted_e2v_table[i])
        for i in range(2, len(sorted_e2v_table), 3):
            new_sorted_e2v_table.append(sorted_e2v_table[i])
        return np.array(new_sorted_e2v_table)
    else:
        return np.array(sorted_e2v_table)


np.set_printoptions(threshold=np.inf)  # type: ignore [arg-type]

sorted_e2v = sort_e2v(grid.get_offset_provider("E2V").table, per_orientation=True)

plot_torus(
    vertice_coords,
    sorted_e2v,
)