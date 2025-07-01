from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import numpy as np
import time
import netCDF4
import os
from pathlib import Path
import glob

def get_torus_cartesian_dimensions(filename):
    nc = netCDF4.Dataset(filename, mode="r")
    sorted_y_coordinates = np.sort(nc["cartesian_y_vertices"][:])
    longitude_dimension = np.count_nonzero(sorted_y_coordinates == 0.0)
    latitude_dimension = int(len(sorted_y_coordinates) / longitude_dimension)
    return (longitude_dimension, latitude_dimension)

def init_grid_manager(
    fname, num_levels=65, transformation=ToZeroBasedIndexTransformation()
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

def get_all_table_names(grid):
    """Extract all available table names from the grid object"""
    table_names = []
    
    # Try to get all offset providers
    try:
        # This might vary depending on the exact API of your grid object
        # You may need to adjust this based on the grid object's attributes
        if hasattr(grid, '_offset_providers'):
            table_names = list(grid._offset_providers.keys())
        elif hasattr(grid, 'offset_providers'):
            table_names = list(grid.offset_providers.keys())
        else:
            # Try some common table names if we can't auto-discover
            common_names = [
                'C2E', 'E2C', 'V2C', 'C2V', 'E2V', 'V2E',
                'C2E2C', 'E2C2E', 'V2C2V', 'C2V2C', 'E2V2E', 'V2E2V',
                'C2CE', 'E2EC', 'V2CV', 'C2VC', 'E2VE', 'V2EV'
            ]
            table_names = []
            for name in common_names:
                try:
                    grid.get_offset_provider(name)
                    table_names.append(name)
                except:
                    pass
    except Exception as e:
        print(f"Error getting table names: {e}")
        table_names = []
    
    return table_names

def get_offset_table(offset_provider):
    """Extract the actual array from an offset provider"""
    if hasattr(offset_provider, "table"):
        return offset_provider.table
    elif hasattr(offset_provider, "_ndarray"):
        return offset_provider._ndarray
    else:
        raise TypeError(f"Does not know how to get the offset table from '{type(offset_provider).__name__}'.")

def save_table_to_file(grid, table_name, output_dir):
    """Save a specific table to a text file"""
    try:
        offset_provider = grid.get_offset_provider(table_name)
        
        # Get the actual array data using the helper function
        table_data = get_offset_table(offset_provider)
        
        filename = output_dir / f"{table_name}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"--- Table '{table_name}' ---\n\n")
            
            if isinstance(table_data, np.ndarray):
                # Save the array data
                if table_data.ndim == 1:
                    np.savetxt(f, table_data, fmt='%d')
                else:
                    np.savetxt(f, table_data, fmt='%d', delimiter='\t')
            else:
                f.write(str(table_data))
        
        print(f"  Saved table '{table_name}' to {filename}")
        return True
    except Exception as e:
        print(f"  Error saving table '{table_name}': {e}")
        return False

def process_grid_file(grid_file_path, base_folder):
    """Process a single grid file and save its tables"""
    grid_file = Path(grid_file_path)
    print(f"\nProcessing: {grid_file.name}")
    
    # Create subfolder for this grid file (without extension)
    output_dir = base_folder / grid_file.stem
    output_dir.mkdir(exist_ok=True)
    print(f"  Created directory: {output_dir}")
    
    file_start = time.perf_counter()
    
    try:
        # Load the torus grid
        print("  Loading grid...")
        torus_grid = get_torus_grid(str(grid_file), 1, ToZeroBasedIndexTransformation())
        
        # Get all available table names
        print("  Discovering available tables...")
        table_names = get_all_table_names(torus_grid)
        
        if not table_names:
            print("  No tables found or couldn't auto-discover tables.")
            return 0, 0
        
        print(f"  Found {len(table_names)} tables: {', '.join(table_names)}")
        
        # Save each table to a separate file
        successful_saves = 0
        for table_name in table_names:
            if save_table_to_file(torus_grid, table_name, output_dir):
                successful_saves += 1
        
        # Save grid dimensions info
        try:
            dimensions = get_torus_cartesian_dimensions(str(grid_file))
            info_file = output_dir / "grid_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Grid file: {grid_file.name}\n")
                f.write(f"Dimensions: {dimensions}\n")
                f.write(f"Tables saved: {successful_saves}/{len(table_names)}\n")
            print(f"  Saved grid info to {info_file}")
        except Exception as e:
            print(f"  Error saving grid info: {e}")
        
        file_end = time.perf_counter()
        print(f"  File processed in {file_end - file_start:.2f} seconds")
        print(f"  Successfully saved {successful_saves}/{len(table_names)} tables")
        
        return successful_saves, len(table_names)
        
    except Exception as e:
        print(f"  Error processing {grid_file.name}: {e}")
        return 0, 0

def main():
    # Prompt user for input folder
    input_folder = input("Path to folder containing grid files: ").strip()
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return
    
    # Find all grid files (assuming .nc extension, but you can modify this)
    file_patterns = ['*.nc', '*.NC', '*.netcdf', '*.NETCDF']
    grid_files = []
    
    for pattern in file_patterns:
        grid_files.extend(input_path.glob(pattern))
    
    if not grid_files:
        print(f"No grid files found in '{input_folder}'")
        print("Looking for files with extensions: .nc, .NC, .netcdf, .NETCDF")
        
        # Allow user to specify custom pattern
        custom_pattern = input("Enter custom file pattern (e.g., '*.dat') or press Enter to exit: ").strip()
        if custom_pattern:
            grid_files = list(input_path.glob(custom_pattern))
        
        if not grid_files:
            print("No files found. Exiting.")
            return
    
    print(f"Found {len(grid_files)} grid files:")
    for gf in grid_files:
        print(f"  - {gf.name}")
    
    # Confirm before processing
    confirm = input(f"\nProcess all {len(grid_files)} files? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    start_time = time.perf_counter()
    
    total_successful = 0
    total_tables = 0
    
    # Process each grid file
    for grid_file in grid_files:
        successful, total = process_grid_file(grid_file, input_path)
        total_successful += successful
        total_tables += total
    
    end_time = time.perf_counter()
    
    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"Processed {len(grid_files)} grid files")
    print(f"Total tables saved: {total_successful}/{total_tables}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Output location: {input_path}")

if __name__ == "__main__":
    main()