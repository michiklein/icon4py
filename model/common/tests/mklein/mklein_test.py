import numpy as np
import time
import netCDF4
import os
import sys

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import gt4py.next as gtx
from stencils_combined import *
from gt4py.next import Dimension

def reorder_edges_by_type(edges, _):
    type_order = ['east', 'north', 'southeast']
    type_buckets = {t: [] for t in type_order}

    for i, edge in enumerate(edges):
        edge_type = type_order[i % 3]
        type_buckets[edge_type].append(edge)

    reordered_edges = []
    for t in type_order:
        reordered_edges.extend(type_buckets[t])

    return reordered_edges

def reorder_cells_by_type(cells, _):
    type_order = ['up', 'down']
    type_buckets = {t: [] for t in type_order}

    for i, cell in enumerate(cells):
        cell_type = type_order[i % 2]
        type_buckets[cell_type].append(cell)

    reordered_cells = []
    for t in type_order:
        reordered_cells.extend(type_buckets[t])

    return reordered_cells

def reorder_trimmed_edges_and_cells(vertices, edges, cells):
    edges_reordered = reorder_edges_by_type(edges, None)
    cells_reordered = reorder_cells_by_type(cells, None)
    return np.array(vertices), np.array(edges_reordered), np.array(cells_reordered)

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
    start = time.time()
    for cid in c_idx:
        edges = grid.get_offset_provider("C2E").ndarray[cid]
        coords = edge_coords[edges]
        idx = sorted(range(3), key=lambda i: (coords[i][1], coords[i][0]))
        top = idx[0] if coords[idx[0]][1] != coords[idx[1]][1] else (idx[0] if coords[idx[0]][0] < coords[idx[1]][0] else idx[1])
        bottom = idx[2] if coords[idx[1]][1] != coords[idx[2]][1] else (idx[1] if coords[idx[1]][0] < coords[idx[2]][0] else idx[2])
        third = next(i for i in range(3) if i not in [top, bottom])
        grid.get_offset_provider("C2E").ndarray[cid] = [edges[top], edges[bottom], edges[third]]

        verts = grid.get_offset_provider("C2V").ndarray[cid]
        coords = vertex_coords[verts]
        idx = sorted(range(3), key=lambda i: (coords[i][1], coords[i][0]))
        top = idx[0] if coords[idx[0]][1] != coords[idx[1]][1] else (idx[0] if coords[idx[0]][0] < coords[idx[1]][0] else idx[1])
        bottom = idx[2] if coords[idx[1]][1] != coords[idx[2]][1] else (idx[1] if coords[idx[1]][0] < coords[idx[2]][0] else idx[2])
        third = next(i for i in range(3) if i not in [top, bottom])
        grid.get_offset_provider("C2V").ndarray[cid] = [verts[top], verts[bottom], verts[third]]
    end = time.time()
    print(f"c2x reorder time: {end - start:.4f} seconds")

def reorder_e2x(grid, grid_file, e_idx):
    vertex_coords = get_coords_v(grid_file)
    cell_coords = get_coords_c(grid_file)
    start = time.time()
    for eid in e_idx:
        verts = grid.get_offset_provider("E2V").ndarray[eid]
        coords = vertex_coords[verts]
        if coords[0][1] < coords[1][1] or (coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]):
            grid.get_offset_provider("E2V").ndarray[eid] = [verts[1], verts[0]]
        else:
            grid.get_offset_provider("E2V").ndarray[eid] = [verts[0], verts[1]]

        cells = grid.get_offset_provider("E2C").ndarray[eid]
        coords = cell_coords[cells]
        if coords[0][1] < coords[1][1] or (coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]):
            grid.get_offset_provider("E2C").ndarray[eid] = [cells[1], cells[0]]
        else:
            grid.get_offset_provider("E2C").ndarray[eid] = [cells[0], cells[1]]
    end = time.time()
    print(f"e2x reorder time: {end - start:.4f} seconds")

def reorder_v2x(grid, grid_file, v_idx):
    vertex_coords = get_coords_v(grid_file)
    edge_coords = get_coords_e(grid_file)
    cell_coords = get_coords_c(grid_file)
    start = time.time()
    for vid in v_idx:
        center = vertex_coords[vid]
        neighbors = grid.get_offset_provider("V2C").ndarray[vid]
        coords = cell_coords[neighbors]
        rel = coords - center
        angles = np.arctan2(-rel[:, 0], -rel[:, 1])
        order = np.argsort(angles)
        grid.get_offset_provider("V2C").ndarray[vid] = neighbors[order]

        neighbors = grid.get_offset_provider("V2E").ndarray[vid]
        coords = edge_coords[neighbors]
        rel = coords - center
        angles = np.arctan2(-rel[:, 0], -rel[:, 1])
        order = np.argsort(angles)
        grid.get_offset_provider("V2E").ndarray[vid] = neighbors[order]
    end = time.time()
    print(f"v2x reorder time: {end - start:.4f} seconds")

def init_grid_manager(fname, num_levels=1, transformation=ToZeroBasedIndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridConfig(num_levels))
    grid_manager(None)
    return grid_manager

def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    return grid_manager.grid

PROGRAMS = {
    "c2e2c": c2e2c_sum_program,
    "c2e2v": c2e2v_sum_program,
    "c2v2c": c2v2c_sum_program,
    "c2v2e": c2v2e_sum_program,
    "e2c2e": e2c2e_sum_program,
    "e2c2v": e2c2v_sum_program,
    "e2v2c": e2v2c_sum_program,
    "e2v2e": e2v2e_sum_program,
    "v2c2e": v2c2e_sum_program,
    "v2c2v": v2c2v_sum_program,
    "v2e2c": v2e2c_sum_program,
    "v2e2v": v2e2v_sum_program,
}

def neighbor_sums(grid, v_idx, e_idx, c_idx):
    os.makedirs("results", exist_ok=True)
    appendix = input("Enter filename appendix (e.g., 'test1'): ").strip()
    include_details = input("Include per-index results? (y/n): ").strip().lower() == "y"
    filename = f"results/neighbor_sums_{appendix or 'default'}.txt"

    id_sets = {"V": v_idx, "E": e_idx, "C": c_idx}
    tables = ["V2C", "V2E", "E2C", "E2V", "C2E", "C2V"]
    rng = np.random.default_rng(42)
    value_map = {
        "V": rng.random(grid.num_vertices),
        "E": rng.random(grid.num_edges),
        "C": rng.random(grid.num_cells),
    }
    domain_map = {
        "V": gtx.domain({Dimension("Vertex"): grid.num_vertices}),
        "E": gtx.domain({Dimension("Edge"): grid.num_edges}),
        "C": gtx.domain({Dimension("Cell"): grid.num_cells}),
    }

    output_lines = []
    timing_summary = []

    for first in tables:
        for second in tables:
            if first[2] != second[0]:
                continue
            combo_key = f"{first.lower()}2{second[2].lower()}"
            program = PROGRAMS[combo_key]
            base_ids = id_sets[first[0]]
            if base_ids.size == 0:
                continue

            input_name = {"V": "vertex_input", "E": "edge_input", "C": "cell_input"}[second[2]]
            output_name = {"V": "vertex_out", "E": "edge_out", "C": "cell_out"}[first[0]]

            input_field = gtx.as_field(domain_map[second[2]], value_map[second[2]])
            result_field = gtx.zeros(domain_map[first[0]])

            start = time.perf_counter()
            program(**{
                input_name: input_field,
                output_name: result_field,
                "offset_provider": grid.offset_providers,
            })
            elapsed = time.perf_counter() - start

            result = result_field.asnumpy()
            timing_summary.append(f"{first}->{second}: {elapsed:.6f}s")

            if include_details:
                output_lines.append(f"{first} -> {second} ({elapsed:.6f}s)")
                output_lines.extend(f"{idx}: {result[idx]:.6f}" for idx in base_ids)
                output_lines.append("")

    with open(filename, "w") as f:
        f.write("Summary of neighbor combinations and timings:\n")
        f.write("\n".join(timing_summary) + "\n\n")
        f.write("\n".join(output_lines))

    print(f"Results written to {filename}")

# --- Now immediately your original code, no __main__ wrapper:

grid_file = "../all_torus_files/torus_100000_100000_512.nc"
grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())

vertices, edges, cells = trim_grid(grid_file)
vertices, edges, cells = reorder_trimmed_edges_and_cells(vertices, edges, cells)

reorder_c2x(grid, grid_file, cells)
reorder_e2x(grid, grid_file, edges)
reorder_v2x(grid, grid_file, vertices)

neighbor_sums(grid, vertices, edges, cells)
