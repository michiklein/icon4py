from icon4py.model.common.grid.grid_manager import (  # type: ignore [import-not-found]
    GridManager,
    IndexTransformation,
    ToZeroBasedIndexTransformation,
)

from icon4py.model.common.grid.vertical import VerticalGridConfig  # type: ignore [import-not-found]
def init_grid_manager(
    fname,
    num_levels=65,
    transformation=ToZeroBasedIndexTransformation(),
):
    grid_manager = GridManager(
        transformation,
        fname,
        VerticalGridConfig(num_levels),
    )
    grid_manager(None)
    return grid_manager


def get_torus_grid(filename, num_levels, transformation):
    grid_manager = init_grid_manager(
        filename, num_levels, transformation
    )
    simple_grid = grid_manager.grid
    return simple_grid

torus_grid = get_torus_grid("/Users/michaelklein/Documents/MASTERTHESIS/all_torus_files/torus_100000_100000_16384.nc", 1, ToZeroBasedIndexTransformation())
print(torus_grid.get_offset_provider("C2E").table)