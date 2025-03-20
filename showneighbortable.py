from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import time



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


# Prompt user for grid file path
grid_file = input("Path to grid file: ")

# Suggest available tables in the prompt
table_name = input(f"Choose table: ")

# Load the torus grid
torus_grid = get_torus_grid(grid_file, 1, ToZeroBasedIndexTransformation())

start = time.perf_counter()
# Print the chosen table
try:
    print(f"\n--- Table '{table_name}' ---\n")
    print(torus_grid.get_offset_provider(table_name).table)
except AttributeError:
    print(f"Error: Table '{table_name}' not found in the grid file.")
end = time.perf_counter()
print(f"Time taken: {end - start} seconds")
