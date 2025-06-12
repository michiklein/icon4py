#MKLEIN MASTER THESIS
import numpy as np
import netCDF4
import os
import sys
import time
from icon4py.model.common.grid.grid_manager import (
    GridManager,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig


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


def init_grid_manager(fname, num_levels=1, transformation=ToZeroBasedIndexTransformation()):
    grid_manager = GridManager(transformation, fname, VerticalGridConfig(num_levels))
    grid_manager(None)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(filename, num_levels, transformation)
    return grid_manager.grid


def reorder_c2x(grid, grid_file):
    # Sort edges/vertices by (y, x) coordinates, then arrange as [top, bottom, third]
    edge_coords = get_coords_e(grid_file)
    vertex_coords = get_coords_v(grid_file)
    start = time.time()
    
    num_cells = grid.num_cells
    c2e_orig = grid.get_offset_provider("C2E").ndarray
    c2v_orig = grid.get_offset_provider("C2V").ndarray
    
    print("Reordering C2E...")
    c2e_start = time.time()
    all_edge_coords = edge_coords[c2e_orig]
    sort_keys = all_edge_coords[:, :, 1] * 1e6 + all_edge_coords[:, :, 0]
    sort_indices = np.argsort(sort_keys, axis=1)
    
    batch_indices = np.arange(num_cells)[:, np.newaxis]
    sorted_coords = all_edge_coords[batch_indices, sort_indices]
    
    y_diff_01 = sorted_coords[:, 0, 1] != sorted_coords[:, 1, 1]
    x_smaller_01 = sorted_coords[:, 0, 0] < sorted_coords[:, 1, 0]
    top_choice = np.where(y_diff_01, 0, np.where(x_smaller_01, 0, 1))
    
    y_diff_12 = sorted_coords[:, 1, 1] != sorted_coords[:, 2, 1]
    x_smaller_12 = sorted_coords[:, 1, 0] < sorted_coords[:, 2, 0]
    bottom_choice = np.where(y_diff_12, 2, np.where(x_smaller_12, 1, 2))
    
    third_choice = 3 - top_choice - bottom_choice
    final_order = np.column_stack([top_choice, bottom_choice, third_choice])
    c2e_new = c2e_orig[batch_indices, sort_indices][batch_indices, final_order]
    grid.get_offset_provider("C2E").ndarray[:] = c2e_new
    c2e_end = time.time()
    
    print("Reordering C2V...")
    c2v_start = time.time()
    all_vertex_coords = vertex_coords[c2v_orig]
    sort_keys_v = all_vertex_coords[:, :, 1] * 1e6 + all_vertex_coords[:, :, 0]
    sort_indices_v = np.argsort(sort_keys_v, axis=1)
    sorted_coords_v = all_vertex_coords[batch_indices, sort_indices_v]
    
    y_diff_01_v = sorted_coords_v[:, 0, 1] != sorted_coords_v[:, 1, 1]
    x_smaller_01_v = sorted_coords_v[:, 0, 0] < sorted_coords_v[:, 1, 0]
    top_choice_v = np.where(y_diff_01_v, 0, np.where(x_smaller_01_v, 0, 1))
    
    y_diff_12_v = sorted_coords_v[:, 1, 1] != sorted_coords_v[:, 2, 1]
    x_smaller_12_v = sorted_coords_v[:, 1, 0] < sorted_coords_v[:, 2, 0]
    bottom_choice_v = np.where(y_diff_12_v, 2, np.where(x_smaller_12_v, 1, 2))
    
    third_choice_v = 3 - top_choice_v - bottom_choice_v
    final_order_v = np.column_stack([top_choice_v, bottom_choice_v, third_choice_v])
    c2v_new = c2v_orig[batch_indices, sort_indices_v][batch_indices, final_order_v]
    grid.get_offset_provider("C2V").ndarray[:] = c2v_new
    c2v_end = time.time()
    
    end = time.time()
    print(f"c2x reorder time: {end - start:.4f} seconds")


def reorder_e2x(grid, grid_file):
    # Order pairs so first element has higher y, or same y with lower x
    vertex_coords = get_coords_v(grid_file)
    cell_coords = get_coords_c(grid_file)
    start = time.time()
    
    e2v_orig = grid.get_offset_provider("E2V").ndarray
    e2c_orig = grid.get_offset_provider("E2C").ndarray
    
    print("Reordering E2V...")
    verts_all = e2v_orig
    coords_all = vertex_coords[verts_all]
    
    y_diff = coords_all[:, 0, 1] - coords_all[:, 1, 1]
    x_diff = coords_all[:, 0, 0] - coords_all[:, 1, 0]
    
    swap_mask = (y_diff < 0) | ((y_diff == 0) & (x_diff > 0))
    
    e2v_new = np.zeros_like(e2v_orig)
    e2v_new[~swap_mask] = verts_all[~swap_mask]
    e2v_new[swap_mask] = verts_all[swap_mask][:, [1, 0]]
    
    print("Reordering E2C...")
    cells_all = e2c_orig
    coords_all = cell_coords[cells_all]
    
    y_diff = coords_all[:, 0, 1] - coords_all[:, 1, 1]
    x_diff = coords_all[:, 0, 0] - coords_all[:, 1, 0]
    
    swap_mask = (y_diff < 0) | ((y_diff == 0) & (x_diff > 0))
    
    e2c_new = np.zeros_like(e2c_orig)
    e2c_new[~swap_mask] = cells_all[~swap_mask]
    e2c_new[swap_mask] = cells_all[swap_mask][:, [1, 0]]
    
    grid.get_offset_provider("E2V").ndarray[:] = e2v_new
    grid.get_offset_provider("E2C").ndarray[:] = e2c_new
    
    end = time.time()
    print(f"e2x reorder time: {end - start:.4f} seconds")


def reorder_v2x(grid, grid_file):
    # Sort neighbors by angle around vertex center (counterclockwise)
    vertex_coords = get_coords_v(grid_file)
    edge_coords = get_coords_e(grid_file)
    cell_coords = get_coords_c(grid_file)
    start = time.time()
    
    num_vertices = grid.num_vertices
    v2c_orig = grid.get_offset_provider("V2C").ndarray
    v2e_orig = grid.get_offset_provider("V2E").ndarray
    
    v2c_new = np.zeros_like(v2c_orig)
    v2e_new = np.zeros_like(v2e_orig)
    
    print("Reordering V2C...")
    chunk_size = min(10000, num_vertices)
    for start_idx in range(0, num_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_vertices)
        chunk_vertices = np.arange(start_idx, end_idx)
        
        centers = vertex_coords[chunk_vertices]
        neighbors = v2c_orig[chunk_vertices]
        neighbor_coords = cell_coords[neighbors]
        rel_coords = neighbor_coords - centers[:, np.newaxis, :]
        angles = np.arctan2(-rel_coords[:, :, 0], rel_coords[:, :, 1])
        sort_indices = np.argsort(angles, axis=1)
        
        for i, vid in enumerate(chunk_vertices):
            v2c_new[vid] = neighbors[i][sort_indices[i]]
    
    print("Reordering V2E...")
    for start_idx in range(0, num_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_vertices)
        chunk_vertices = np.arange(start_idx, end_idx)
        
        centers = vertex_coords[chunk_vertices]
        neighbors = v2e_orig[chunk_vertices]
        neighbor_coords = edge_coords[neighbors]
        rel_coords = neighbor_coords - centers[:, np.newaxis, :]
        angles = np.arctan2(-rel_coords[:, :, 0], rel_coords[:, :, 1])
        sort_indices = np.argsort(angles, axis=1)
        
        for i, vid in enumerate(chunk_vertices):
            v2e_new[vid] = neighbors[i][sort_indices[i]]
    
    grid.get_offset_provider("V2C").ndarray[:] = v2c_new
    grid.get_offset_provider("V2E").ndarray[:] = v2e_new
    
    end = time.time()
    print(f"v2x reorder time: {end - start:.4f} seconds")


def save_reordered_grid(grid, original_file, output_file):
    with netCDF4.Dataset(original_file, 'r') as src:
        with netCDF4.Dataset(output_file, 'w') as dst:
            for name, dimension in src.dimensions.items():
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
            
            for name, variable in src.variables.items():
                dst.createVariable(name, variable.datatype, variable.dimensions)
                dst[name][:] = src[name][:]
                for attr in variable.ncattrs():
                    dst[name].setncattr(attr, variable.getncattr(attr))
            
            for attr in src.ncattrs():
                dst.setncattr(attr, src.getncattr(attr))
    
    with netCDF4.Dataset(output_file, 'a') as nc:
        connectivity_mapping = {}
        
        possible_names = {
            'C2V': ['vertex_of_cell', 'vertices_of_cell', 'cell_vertex_index'],
            'C2E': ['edge_of_cell', 'edges_of_cell', 'cell_edge_index'],
            'E2V': ['vertex_of_edge', 'vertices_of_edge', 'edge_vertex_index'],
            'E2C': ['cell_of_edge', 'cells_of_edge', 'edge_cell_index'],
            'V2E': ['edge_of_vertex', 'edges_of_vertex', 'vertex_edge_index'],
            'V2C': ['cell_of_vertex', 'cells_of_vertex', 'vertex_cell_index']
        }
        
        for grid_name, possible in possible_names.items():
            for name in possible:
                if name in nc.variables:
                    connectivity_mapping[name] = grid_name
                    break
        
        for nc_var_name, grid_name in connectivity_mapping.items():
            connectivity_data = grid.get_offset_provider(grid_name).ndarray
            if nc[nc_var_name].shape != connectivity_data.shape:
                connectivity_data = connectivity_data.T
            nc[nc_var_name][:] = connectivity_data


def reorder_grid_file(input_file, output_file=None):
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_reordered.nc"
    
    grid = get_torus_grid(input_file, 1, ToZeroBasedIndexTransformation())
    
    reorder_c2x(grid, input_file)
    reorder_e2x(grid, input_file)
    reorder_v2x(grid, input_file)
    
    save_reordered_grid(grid, input_file, output_file)
    
    return output_file


if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        import glob
        import time
        
        folder = sys.argv[1]
        nc_files = glob.glob(os.path.join(folder, "*.nc"))
        nc_files = [f for f in nc_files if not f.endswith('_reordered.nc')]
        
        nc_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
        
        total_start = time.time()
        print(f"Processing {len(nc_files)} files...")
        
        for i, nc_file in enumerate(nc_files, 1):
            basename = os.path.basename(nc_file)
            reordered_file = nc_file.replace('.nc', '_reordered.nc')
            
            if os.path.exists(reordered_file):
                print(f"[{i}/{len(nc_files)}] Skipping {basename} (already exists)")
                continue
            
            print(f"[{i}/{len(nc_files)}] Processing {basename}")
            
            file_start = time.time()
            try:
                reorder_grid_file(nc_file)
                elapsed = time.time() - file_start
                print(f"  Completed in {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - file_start
                print(f"  FAILED after {elapsed:.1f}s: {e}")
        
        total_elapsed = time.time() - total_start
        print(f"Total time: {total_elapsed:.1f}s")
        
    elif len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        try:
            reorder_grid_file(input_file, output_file)
        except Exception as e:
            print(f"Error processing file: {e}")
            sys.exit(1)
    else:
        print("Usage: python grid_reorder.py <input_file.nc> [output_file.nc]")
        print("   or: python grid_reorder.py <folder_path>")
        sys.exit(1)