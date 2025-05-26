import numpy as np
import cupy as cp
import time
import netCDF4
import os
import sys
import re

from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    ToZeroBasedIndexTransformation,
    _v2e2v_connectivity,
    _v2e2c_connectivity,
    _v2c2e_connectivity,
    _e2v2e_connectivity,
    _e2v2c_connectivity,
    _e2c2e_connectivity,
    _e2c2v_connectivity,
    _c2v2e_connectivity,
    _c2v2c_connectivity,
    _c2e2v_connectivity,
    _c2e2c_connectivity,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import gt4py.next as gtx
from stencils_combined import *
from gt4py.next import Dimension

xp = cp if "gpu" in str(b_end).lower() else np #gpu or cpu?

def reorder_edges_by_type(edges, vertex_coords, grid):
    # Define final order of edge types
    type_order = ["east", "north", "southeast"]
    type_buckets = {t: [] for t in type_order}
    
    # We need at least 2 edges to determine the pattern
    if len(edges) >= 2:
        # Analyze first two edges
        first_edge = int(edges[0])  # Convert to int
        second_edge = int(edges[1])  # Convert to int
        
        # Get vertex coordinates for first edge
        v1_1, v1_2 = grid.get_offset_provider("E2V").ndarray[first_edge]
        
        # Convert indices to int to avoid CuPy-NumPy conversion issues
        v1_1_int = int(v1_1)
        v1_2_int = int(v1_2)
        coords1_1, coords1_2 = vertex_coords[v1_1_int], vertex_coords[v1_2_int]
        
        dx1, dy1 = coords1_2[0] - coords1_1[0], coords1_2[1] - coords1_1[1]
        
        # Get vertex coordinates for second edge
        v2_1, v2_2 = grid.get_offset_provider("E2V").ndarray[second_edge]
        
        # Convert indices to int
        v2_1_int = int(v2_1)
        v2_2_int = int(v2_2)
        coords2_1, coords2_2 = vertex_coords[v2_1_int], vertex_coords[v2_2_int]
        
        dx2, dy2 = coords2_2[0] - coords2_1[0], coords2_2[1] - coords2_1[1]
        
        # Classify first edge
        if abs(dx1) > abs(dy1):  # More horizontal
            first_type = "east"
        elif dx1 > 0 and dy1 > 0:  # Diagonal up-right
            first_type = "southeast"
        else:  # Mostly vertical
            first_type = "north"
            
        # Classify second edge
        if abs(dx2) > abs(dy2):  # More horizontal
            second_type = "east"
        elif dx2 > 0 and dy2 > 0:  # Diagonal up-right
            second_type = "southeast"
        else:  # Mostly vertical
            second_type = "north"
        
        # Determine the original sequence (how edges appear in the input)
        original_sequence = []
        original_sequence.append(first_type)
        original_sequence.append(second_type)
        
        # Predict the third type (the one missing from the first two)
        remaining_type = [t for t in type_order if t not in [first_type, second_type]]
        if len(remaining_type) == 1:
            third_type = remaining_type[0]
        else:
            # If first and second are the same, predict based on standard pattern
            third_type = type_order[(type_order.index(first_type) + 2) % 3]
        
        original_sequence.append(third_type)
        
        # Now classify each edge by its position in the sequence
        for i, edge in enumerate(edges):
            edge_type = original_sequence[i % 3]
            edge_int = int(edge)  # Convert to int
            type_buckets[edge_type].append(edge_int)
    else:
        # If not enough edges, fall back to the original method
        for i, edge in enumerate(edges):
            edge_type = type_order[i % 3]
            edge_int = int(edge)  # Convert to int
            type_buckets[edge_type].append(edge_int)
    
    # Combine in the specified order - always ["east", "north", "southeast"]
    reordered_edges = []
    for t in type_order:
        reordered_edges.extend(type_buckets[t])
        
    return reordered_edges

def reorder_cells_by_type(cells, vertex_coords, grid):
    type_order = ["up", "down"]
    type_buckets = {t: [] for t in type_order}
    
    # Determine type of first cell based on vertex coordinates
    if len(cells) > 0:
        first_cell = int(cells[0])  # Convert to int
        vertices = grid.get_offset_provider("C2V").ndarray[first_cell]
        
        # Convert vertices to integers for indexing into NumPy array
        vertices_int = [int(v) for v in vertices]
        coords = [vertex_coords[v] for v in vertices_int]
        
        # Calculate center point
        center_x = sum(c[0] for c in coords) / len(coords)
        center_y = sum(c[1] for c in coords) / len(coords)
        
        # Find topmost vertex
        top_idx = max(range(len(coords)), key=lambda i: coords[i][1])
        
        # If topmost vertex is above center, it's pointing up
        is_up = coords[top_idx][1] > center_y
        starting_type = "up" if is_up else "down"
        start_idx = type_order.index(starting_type)
        
        # Assign cells to buckets based on the established pattern
        for i, cell in enumerate(cells):
            cell_type = type_order[(start_idx + i) % 2]
            cell_int = int(cell)  # Convert to int
            type_buckets[cell_type].append(cell_int)
    
    # Combine in the specified order
    reordered_cells = []
    for t in type_order:
        reordered_cells.extend(type_buckets[t])
        
    return reordered_cells

def reorder_trimmed_edges_and_cells(vertices, edges, cells, grid_file, grid):
    # Get vertex coordinates for geometric checks
    vertex_coords = get_coords_v(grid_file)
    
    edges_reordered = reorder_edges_by_type(edges, vertex_coords, grid)
    cells_reordered = reorder_cells_by_type(cells, vertex_coords, grid)
    return xp.array(vertices), xp.array(edges_reordered), xp.array(cells_reordered)


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


def trim_grid(grid_file, grid):
    nx, ny = get_torus_cartesian_dimensions(grid_file)
    halo = 2
    usable_x = nx - 2 * halo
    usable_y = ny - 2 * halo
    base = min(usable_x, usable_y)
    side = (base // 32) * 32 if base >= 32 else (base // 4) * 4
    start_i = (nx - side) // 2
    start_j = (ny - side) // 2
    end_i = start_i + side
    end_j = start_j + side
    
    # First select vertices in the square region
    v_idx = []
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            flat = i * ny + j
            v_idx.append(flat)
    
    v_idx = xp.array(v_idx)
    vertex_set = set(v_idx.get().tolist() if xp.__name__ == "cupy" else v_idx.tolist())
    
    # Find edges that connect selected vertices
    e_idx = []
    for eid in range(grid.num_edges):
        vertices = grid.get_offset_provider("E2V").ndarray[eid]
        if all(v in vertex_set for v in vertices):
            e_idx.append(eid)
    
    # Find cells that connect selected vertices
    c_idx = []
    for cid in range(grid.num_cells):
        vertices = grid.get_offset_provider("C2V").ndarray[cid]
        if all(v in vertex_set for v in vertices):
            c_idx.append(cid)
    
    e_idx = xp.array(e_idx)
    c_idx = xp.array(c_idx)
    
    expected = side**2
    print(f"expected {expected} vertices")
    print(f"Selected {len(v_idx)} vertices, {len(e_idx)} edges, and {len(c_idx)} cells")
    
    return v_idx, e_idx, c_idx


def reorder_c2x(grid, grid_file):
    edge_coords = get_coords_e(grid_file)
    vertex_coords = get_coords_v(grid_file)
    start = time.time()
    for cid in range(grid.num_cells):
        cid_int = int(cid)  # Convert to int
        edges = grid.get_offset_provider("C2E").ndarray[cid_int]
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
        grid.get_offset_provider("C2E").ndarray[cid_int] = [
            edges[top],
            edges[bottom],
            edges[third],
        ]

        verts = grid.get_offset_provider("C2V").ndarray[cid_int]
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
        grid.get_offset_provider("C2V").ndarray[cid_int] = [
            verts[top],
            verts[bottom],
            verts[third],
        ]
    end = time.time()
    print(f"c2x reorder time: {end - start:.4f} seconds")


def reorder_e2x(grid, grid_file):
    vertex_coords = get_coords_v(grid_file)
    cell_coords = get_coords_c(grid_file)
    start = time.time()
    for eid in range(grid.num_edges):
        eid_int = int(eid)  # Convert to int
        verts = grid.get_offset_provider("E2V").ndarray[eid_int]
        coords = vertex_coords[verts]
        if coords[0][1] < coords[1][1] or (
            coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]
        ):
            grid.get_offset_provider("E2V").ndarray[eid_int] = [verts[1], verts[0]]
        else:
            grid.get_offset_provider("E2V").ndarray[eid_int] = [verts[0], verts[1]]

        cells = grid.get_offset_provider("E2C").ndarray[eid_int]
        coords = cell_coords[cells]
        if coords[0][1] < coords[1][1] or (
            coords[0][1] == coords[1][1] and coords[0][0] > coords[1][0]
        ):
            grid.get_offset_provider("E2C").ndarray[eid_int] = [cells[1], cells[0]]
        else:
            grid.get_offset_provider("E2C").ndarray[eid_int] = [cells[0], cells[1]]
    end = time.time()
    print(f"e2x reorder time: {end - start:.4f} seconds")


def reorder_v2x(grid, grid_file):
    vertex_coords = get_coords_v(grid_file)
    edge_coords = get_coords_e(grid_file)
    cell_coords = get_coords_c(grid_file)
    start = time.time()
    for vid in range(grid.num_vertices):
        vid_int = int(vid)  # Convert to int
        center = vertex_coords[vid_int]
        neighbors = grid.get_offset_provider("V2C").ndarray[vid_int]
        coords = cell_coords[neighbors]
        rel = coords - center
        angles = np.arctan2(-rel[:, 0], -rel[:, 1])
        order = np.argsort(angles)
        grid.get_offset_provider("V2C").ndarray[vid_int] = neighbors[order]

        neighbors = grid.get_offset_provider("V2E").ndarray[vid_int]
        coords = edge_coords[neighbors]
        rel = coords - center
        angles = np.arctan2(-rel[:, 0], -rel[:, 1])
        order = np.argsort(angles)
        grid.get_offset_provider("V2E").ndarray[vid_int] = neighbors[order]
    end = time.time()
    print(f"v2x reorder time: {end - start:.4f} seconds")

def reindex_edges(grid, e_idx):
    edge_map = {int(old): new for new, old in enumerate(e_idx)}

    for name in ["C2E", "V2E"]:
        table = grid.get_offset_provider(name).ndarray
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = table[i, j]
                if val in edge_map:
                    table[i, j] = edge_map[val]

    for name in ["E2V", "E2C"]:
        table = grid.get_offset_provider(name).ndarray
        new_table = table.copy()
        swapped = set()

        for old_id, new_id in edge_map.items():
            if old_id == new_id or old_id in swapped or new_id in swapped:
                continue
            new_table[old_id], new_table[new_id] = table[new_id], table[old_id]
            swapped.update((old_id, new_id))

        table[...] = new_table

def reindex_cells(grid, c_idx):
    cell_map = {int(old): new for new, old in enumerate(c_idx)}

    for name in ["E2C", "V2C"]:
        table = grid.get_offset_provider(name).ndarray
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = table[i, j]
                if val in cell_map:
                    table[i, j] = cell_map[val]

    for name in ["C2E", "C2V"]:
        table = grid.get_offset_provider(name).ndarray
        new_table = table.copy()
        swapped = set()

        for old_id, new_id in cell_map.items():
            if old_id == new_id or (old_id in swapped or new_id in swapped):
                continue

            new_table[old_id], new_table[new_id] = table[new_id], table[old_id]
            swapped.update({old_id, new_id})

        table[...] = new_table
        
def reindex_vertices(grid, v_idx):
    vertex_map = {int(old): new for new, old in enumerate(v_idx)}

    for name in ["E2V", "C2V"]:
        table = grid.get_offset_provider(name).ndarray
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = table[i, j]
                if val in vertex_map:
                    table[i, j] = vertex_map[val]

    for name in ["V2E", "V2C"]:
        table = grid.get_offset_provider(name).ndarray
        new_table = table.copy()
        swapped = set()

        for old_id, new_id in vertex_map.items():
            if old_id == new_id or old_id in swapped or new_id in swapped:
                continue

            new_table[old_id], new_table[new_id] = table[new_id], table[old_id]
            swapped.update((old_id, new_id))

        table[...] = new_table


def init_grid_manager(
    fname, num_levels=1, transformation=ToZeroBasedIndexTransformation()
):
    grid_manager = GridManager(transformation, fname, VerticalGridConfig(num_levels))
    grid_manager(None)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    return grid_manager.grid

def overwrite_chained_connectivities(grid):
    v2e = grid.get_offset_provider("V2E").ndarray
    v2c = grid.get_offset_provider("V2C").ndarray
    e2v = grid.get_offset_provider("E2V").ndarray
    e2c = grid.get_offset_provider("E2C").ndarray
    c2v = grid.get_offset_provider("C2V").ndarray
    c2e = grid.get_offset_provider("C2E").ndarray
    
    grid.connectivities["V2E2V"] = _v2e2v_connectivity(v2e, e2v)
    grid.connectivities["V2E2C"] = _v2e2c_connectivity(v2c)
    grid.connectivities["V2C2E"] = _v2c2e_connectivity(v2c, c2e)
    grid.connectivities["V2C2V"] = _v2e2v_connectivity(v2c, c2v)
    grid.connectivities["E2V2E"] = _e2v2e_connectivity(e2v, v2e)
    grid.connectivities["E2V2C"] = _e2v2c_connectivity(e2v, v2c)
    grid.connectivities["E2C2E"] = _e2c2e_connectivity(e2c, c2e)
    grid.connectivities["E2C2V"] = _e2c2v_connectivity(e2c, c2v)
    grid.connectivities["C2V2E"] = _c2v2e_connectivity(c2v, v2e)
    grid.connectivities["C2V2C"] = _c2v2c_connectivity(c2v, v2c)
    grid.connectivities["C2E2V"] = _c2e2v_connectivity(c2v)
    grid.connectivities["C2E2C"] = _c2e2c_connectivity(c2e, e2c)
    
    return grid

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

    # Determine backend and device
    backend_str = str(b_end).lower()
    backend_type = "gtfn" if "gtfn" in backend_str else "dace"
    device_type = "gpu" if "gpu" in backend_str else "cpu"

    # Auto-increment file index
    base_filename = f"neighbor_sums_{backend_type}_{device_type}"
    existing_files = os.listdir("results")
    existing_nums = [
        int(match.group(1))
        for fname in existing_files
        if (match := re.match(rf"{re.escape(base_filename)}_(\d+)\.txt", fname))
    ]
    next_num = max(existing_nums, default=0) + 1
    filename = f"results/{base_filename}_{next_num:03d}.txt"

    if xp.__name__ == "cupy":
        xp.get_default_memory_pool().free_all_blocks()
    
    id_sets = {"V": v_idx, "E": e_idx, "C": c_idx}
    tables = ["V2C", "V2E", "E2C", "E2V", "C2E", "C2V"]
    rng = np.random.default_rng(42)
    
    value_map = {
        "V": xp.asarray(rng.random(grid.num_vertices)),
        "E": xp.asarray(rng.random(grid.num_edges)),
        "C": xp.asarray(rng.random(grid.num_cells)),
    }

    
    if xp.__name__ == "cupy":
        for k in value_map:
            value_map[k] = cp.asarray(value_map[k])

    domain_map = {
    "V": gtx.domain({Dimension("Vertex"): grid.num_vertices}),
    "E": gtx.domain({Dimension("Edge"): grid.num_edges}),
    "C": gtx.domain({Dimension("Cell"): grid.num_cells}),
}


    # Rest of the function remains the same
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

            input_name = {"V": "vertex_input", "E": "edge_input", "C": "cell_input"}[
                second[2]
            ]
            output_name = {"V": "vertex_out", "E": "edge_out", "C": "cell_out"}[
                first[0]
            ]

            input_field = gtx.as_field(domain_map[second[2]], value_map[second[2]], allocator=b_end)
            result_field = gtx.zeros(domain_map[first[0]], allocator=b_end)

            start = time.perf_counter()
            program(
                **{
                    input_name: input_field,
                    output_name: result_field,
                    "offset_provider": grid.offset_providers,
                    "num_edges": np.int64(len(e_idx)),
                    "num_cells": np.int64(len(c_idx)),
                }
            )
            elapsed = time.perf_counter() - start

            result = result_field.ndarray
            timing_summary.append(f"{first}->{second}: {elapsed:.6f}s")

            
            output_lines.append(f"{first} -> {second} ({elapsed:.6f}s)")
            num_elements = len(base_ids)
            output_lines.extend(f"{i}: {result[i]:.6f}" for i in range(num_elements))
            output_lines.append("")

    with open(filename, "w") as f:
        f.write(f"Summary of neighbor combinations and timings ({backend_type}, {device_type}):\n")
        f.write("\n".join(timing_summary) + "\n\n")
        f.write("\n".join(output_lines))

    print(f"Results written to {filename}")

if __name__ == "__main__":
    grid_file = "../all_torus_files/torus_100000_100000_1024.nc"
    grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())
    
    vertices, edges, cells = trim_grid(grid_file, grid)
    vertices, edges, cells = reorder_trimmed_edges_and_cells(vertices, edges, cells, grid_file, grid)
    
    reorder_c2x(grid, grid_file)
    reorder_e2x(grid, grid_file)
    reorder_v2x(grid, grid_file)
    reindex_cells(grid, cells)
    reindex_edges(grid, edges)
    reindex_vertices(grid, vertices)
    
    grid = overwrite_chained_connectivities(grid)
    
    if xp.__name__ == "cupy":
        cp.get_default_memory_pool().free_all_blocks() 
        for dim_name, connectivity in grid.connectivities.items():
            if isinstance(connectivity, np.ndarray):
                grid.connectivities[dim_name] = cp.asarray(connectivity)
    
    neighbor_sums(grid, vertices, edges, cells)