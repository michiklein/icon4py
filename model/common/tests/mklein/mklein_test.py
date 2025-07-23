import numpy as np
import cupy as cp
import time
import netCDF4
import os
import sys
import re

from icon4py.model.common.grid.grid_manager import (
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
from icon4py.model.common.grid.vertical import VerticalGridConfig
import gt4py.next as gtx
from stencils_combined import *
from icon4py.model.common.dimension import EdgeDim, VertexDim, CellDim, KDim
from gt4py.next import int32

b_end = gtx.gtfn_gpu
xp = cp if "gpu" in str(b_end).lower() else np

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

def init_grid_manager(fname, num_levels=1, transformation=ToZeroBasedIndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridConfig(num_levels))
    grid_manager(None)
    return grid_manager

def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    return grid_manager.grid

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
    
    v_idx = []
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            flat = i * ny + j
            v_idx.append(flat)
    
    v_idx = xp.array(v_idx)
    vertex_set = set(v_idx.get().tolist() if xp.__name__ == "cupy" else v_idx.tolist())
    
    e_idx = []
    for eid in range(grid.num_edges):
        vertices = grid.get_offset_provider("E2V").ndarray[eid]
        if all(v in vertex_set for v in vertices):
            e_idx.append(eid)
    
    c_idx = []
    for cid in range(grid.num_cells):
        vertices = grid.get_offset_provider("C2V").ndarray[cid]
        if all(v in vertex_set for v in vertices):
            c_idx.append(cid)
    
    e_idx = xp.array(e_idx)
    c_idx = xp.array(c_idx)
    
    expected = side**2
    print(f"Expected {expected} vertices")
    print(f"Selected {len(v_idx)} vertices, {len(e_idx)} edges, and {len(c_idx)} cells")
    
    return v_idx, e_idx, c_idx

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

def neighbor_sums(grid, v_idx, e_idx, c_idx, levels):
    os.makedirs("results", exist_ok=True)

    backend_str = str(b_end).lower()
    backend_type = "gtfn" if "gtfn" in backend_str else "dace"
    device_type = "gpu" if "gpu" in backend_str else "cpu"

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
        "V": xp.asarray(rng.random((grid.num_vertices, levels))),
        "E": xp.asarray(rng.random((grid.num_edges, levels))),
        "C": xp.asarray(rng.random((grid.num_cells, levels))),
    }

    if xp.__name__ == "cupy":
        for k in value_map:
            value_map[k] = cp.asarray(value_map[k])

    domain_map = {
        "V": gtx.domain({VertexDim: grid.num_vertices, KDim: levels}),
        "E": gtx.domain({EdgeDim: grid.num_edges, KDim: levels}),
        "C": gtx.domain({CellDim: grid.num_cells, KDim: levels}),
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

            input_field = gtx.as_field(domain_map[second[2]], value_map[second[2]], allocator=b_end)
            result_field = gtx.zeros(domain_map[first[0]], allocator=b_end)

            start = time.perf_counter()
            program(
                **{
                    input_name: input_field,
                    output_name: result_field,
                    "num_edges": int32(len(e_idx)),
                    "num_cells": int32(len(c_idx)),
                    "offset_provider": grid.offset_providers,
                }
            )
            elapsed = time.perf_counter() - start

            result = result_field.ndarray
            timing_summary.append(f"{first}->{second}: {elapsed:.6f}s")
            
            output_lines.append(f"{first} -> {second} ({elapsed:.6f}s)")
            num_elements = len(base_ids)
            for i in range(num_elements):
                level_0_value = float(result[i, 0])
                output_lines.append(f"{i}: {level_0_value:.6f}")
            output_lines.append("")

    with open(filename, "w") as f:
        f.write(f"Summary of neighbor combinations and timings ({backend_type}, {device_type}):\n")
        f.write("\n".join(timing_summary) + "\n\n")
        f.write("\n".join(output_lines))

    print(f"Results written to {filename}")

if __name__ == "__main__":
    
    grid_file = "../all_torus_files/torus_100000_100000_1024_reordered.nc"
    levels = 80
    grid = get_torus_grid(grid_file, levels, ToZeroBasedIndexTransformation())
    
    vertices, edges, cells = trim_grid(grid_file, grid)
    
    
    reindex_cells(grid, cells)
    reindex_edges(grid, edges)
    reindex_vertices(grid, vertices)
    
    
    if xp.__name__ == "cupy":
        for name, provider in grid.offset_providers.items():
            if hasattr(provider, 'ndarray'):
                grid.offset_providers[name] = gtx.as_connectivity(
                    provider.domain,
                    codomain=provider.codomain, 
                    data=provider.ndarray, 
                    skip_value=-1,
                    allocator=b_end
                )

    neighbor_sums(grid, vertices, edges, cells, levels)