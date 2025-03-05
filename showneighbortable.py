from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]


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


# Define the available neighbor tables
neighbor_tables = [
    "C2E",
    "C2V",
    "E2C",
    "E2V",
    "V2C",
    "V2E",
    "E2C2E",
    "E2C2V",
    "V2E2V",
    "C2E2C",
    "C2E2CO",
    "C2E2C2E",
    "C2E2C2E2C",
    "E2C2E2C",
    "E2C2EO",
    "C2E2EO",
]

# Prompt user for grid file path
grid_file = input("Path to grid file: ")

# Suggest available tables in the prompt
table_name = input(f"Choose table from {', '.join(neighbor_tables)}): ")

# Load the torus grid
torus_grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())

# Print the chosen table
try:
    print(f"\n--- Table '{table_name}' ---\n")
    print(torus_grid.get_offset_provider(table_name).table)
except AttributeError:
    print(f"Error: Table '{table_name}' not found in the grid file.")
