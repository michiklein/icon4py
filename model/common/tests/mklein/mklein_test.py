from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import numpy as np
import time
import netCDF4


def get_torus_cartesian_dimensions(grid):
    nc = netCDF4.Dataset(grid, mode="r")
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
    nx, ny = get_torus_cartesian_dimensions(grid)
    halo = 2

    usable_x = nx - 2 * halo
    usable_y = ny - 2 * halo
    base = min(usable_x, usable_y)
    side = (base // 32) * 32 if base >= 32 else (base // 4) * 4

    start_i = (nx - side) // 2
    start_j = (ny - side) // 2
    end_i = start_i + side
    end_j = start_j + side

    v_idx, e_idx, c_idx = [], [], []

    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            flat = i * ny + j
            v_idx.append(flat)
            e_idx.append(flat)
            c_idx.append(flat)

    expected = side**2
    actual = len(v_idx)
    status = "ok" if actual == expected else "mismatch"
    print(f"{actual} vertices (expected {expected}) - {status}")

    return np.array(v_idx), np.array(e_idx), np.array(c_idx)


def reorder_c2x(grid, grid_file, c_idx):
    edge_coords = get_coords_e(grid_file)
    vertex_coords = get_coords_v(grid_file)

    edge_of_cell = grid.edge_of_cell
    vertex_of_cell = grid.vertex_of_cell

    for cid in c_idx:
        # --- Reorder edge_of_cell ---
        edges = edge_of_cell[cid]
        coords = edge_coords[edges]
        idx = sorted(range(3), key=lambda i: (coords[i][1], coords[i][0]))
        top = (
            idx[0]
            if coords[idx[0]][1] != coords[idx[1]][1]
            else (idx[0] if coords[idx[0]][0] < coords[idx[1]][0] else idx[1])
        )
        bottom = (
            idx[2]
            if coords[idx[1]][1] != coords[idx[2]][1]
            else (idx[1] if coords[idx[1]][0] < coords[idx[2]][0] else idx[2])
        )
        third = next(i for i in range(3) if i not in [top, bottom])
        edge_of_cell[cid] = [edges[top], edges[bottom], edges[third]]

        # --- Reorder vertex_of_cell ---
        verts = vertex_of_cell[cid]
        coords = vertex_coords[verts]
        idx = sorted(range(3), key=lambda i: (coords[i][1], coords[i][0]))
        top = (
            idx[0]
            if coords[idx[0]][1] != coords[idx[1]][1]
            else (idx[0] if coords[idx[0]][0] < coords[idx[1]][0] else idx[1])
        )
        bottom = (
            idx[2]
            if coords[idx[1]][1] != coords[idx[2]][1]
            else (idx[1] if coords[idx[1]][0] < coords[idx[2]][0] else idx[2])
        )
        third = next(i for i in range(3) if i not in [top, bottom])
        vertex_of_cell[cid] = [verts[top], verts[bottom], verts[third]]


def reorder_e2x(grid, grid_file, e_idx):
    vertex_coords = get_coords_v(grid_file)
    cell_coords = get_coords_c(grid_file)

    edge_vertices = grid.edge_vertices
    adjacent_cells = grid.adjacent_cell_of_edge

    for eid in e_idx:
        # --- Reorder edge_vertices: lower vertex first ---
        verts = edge_vertices[eid]
        coords = vertex_coords[verts]
        if coords[0][1] < coords[1][1] or (coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]):
            edge_vertices[eid] = [verts[1], verts[0]]

        # --- Reorder adjacent_cell_of_edge: lower cell first ---
        cells = adjacent_cells[eid]
        coords = cell_coords[cells]
        if coords[0][1] < coords[1][1] or (coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]):
            adjacent_cells[eid] = [cells[1], cells[0]]


def reorder_vertex_connections_in_grid(grid, grid_file, v_idx):
    vertex_coords = get_coords_v(grid_file)
    edge_coords = get_coords_e(grid_file)
    cell_coords = get_coords_c(grid_file)

    cells_of_vertex = grid.cells_of_vertex
    edges_of_vertex = grid.edges_of_vertex

    for vid in v_idx:
        center = vertex_coords[vid]

        # --- Reorder cells_of_vertex ---
        neighbors = cells_of_vertex[vid]
        coords = cell_coords[neighbors]
        rel = coords - center
        angles = np.arctan2(-rel[:, 0], -rel[:, 1])  # angle from top (y axis)
        order = np.argsort(angles)
        grid.cells_of_vertex[vid] = neighbors[order]

        # --- Reorder edges_of_vertex ---
        neighbors = edges_of_vertex[vid]
        coords = edge_coords[neighbors]
        rel = coords - center
        angles = np.arctan2(-rel[:, 0], -rel[:, 1])
        order = np.argsort(angles)
        grid.edges_of_vertex[vid] = neighbors[order]


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
vertices, edges, cells = trim_grid(grid_file)

print(get_coords_v(grid_file)[vertices])
