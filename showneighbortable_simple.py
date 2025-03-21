from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    ToZeroBasedIndexTransformation,
)
from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
import time
from icon4py.model.common.grid import simple


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

# Suggest available tables in the prompt
table_name = input(f"Choose table: ")

start = time.perf_counter()
# Print the chosen table
try:
    print(f"\n--- Table '{table_name}' ---\n")
    print(simple.SimpleGrid().get_offset_provider(table_name).table)
except AttributeError:
    print(f"Error: Table '{table_name}' not found in the grid file.")
end = time.perf_counter()
print(f"Time taken: {end - start} seconds")
