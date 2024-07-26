# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import enum
import logging
import uuid
from typing import Protocol

import gt4py.next as gtx

import icon4py.model.common.decomposition.definitions as defs
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid, icon as icon_grid
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)


# TODO (@halungge) do we need three of those: one for each dimension?
class DecompositionFlag(enum.IntEnum):
    #: cell is owned by this rank
    OWNED = (0,)
    #: cell is in the first halo line: that is cells that share and edge with an owned cell
    FIRST_HALO_LINE = (1,)
    #: cell is in the second halo line: that is cells that share only a vertex with an owned cell (and at least an edge with a FIRST_HALO_LINE cell)
    SECOND_HALO_LINE = 2


SKIP_VALUE = -1


class HaloGenerator:
    """Creates necessary halo information for a given rank."""

    def __init__(
        self,
        run_properties: defs.ProcessProperties,
        rank_mapping: xp.ndarray,
        connectivities:dict,
        num_levels:int,
    ):
        """

        Args:
            run_properties: contains information on the communicator and local compute node.
            rank_mapping: array with shape (global_num_cells,): mapping of global cell indices to their rank in the distribution
            connectivities: connectivity arrays needed to construct the halos
            num_levels: # TODO (@halungge): should not be needed here
        """
        self._props = run_properties
        self._mapping = rank_mapping
        self._connectivities = connectivities
        self._num_levels = num_levels

    @property
    def face_face_connectivity(self):
        return self._connectivity(dims.C2E2CDim)
    
    @property
    def edge_face_connectivity(self):
        return self._connectivity(dims.E2CDim)
    
    @property
    def face_edge_connectivity(self):
        return self._connectivity(dims.C2EDim)


    @property
    def node_edge_connectivity(self):
        return self._connectivity(dims.V2EDim)

    @property
    def node_face_connectivity(self):
        return self._connectivity(dims.V2CDim)
    
    @property   
    def face_node_connectivity(self):
        return self._connectivity(dims.C2VDim)
    
    def _validate(self):
        assert self._mapping.ndim == 1
        # the decomposition should match the communicator size
        assert xp.max(self._mapping) == self._props.comm_size - 1

    def _post_init(self):
        self._validate()

    def _connectivity(self, dim) -> xp.ndarray:
        try:
            conn_table = self._connectivities[dim]
            return conn_table
        except KeyError as err:
            raise (f"Connectivity for dimension {dim} is not available") from err

    def next_halo_line(self, cell_line: xp.ndarray, depot=None):
        """Returns the global indices of the next halo line.

        Args:
            cell_line: global indices of cells we want to find the neighbors of
            depot: global indices that have already been collected
        Returns:
            next_halo_cells: global indices of the next halo line
        """
        cell_neighbors = self._cell_neighbors(cell_line)

        if depot is not None:
            cells_so_far = xp.hstack((depot, cell_line))
        else:
            cells_so_far = cell_line

        next_halo_cells = xp.setdiff1d(xp.unique(cell_neighbors), cells_so_far, assume_unique=True)
        return next_halo_cells

    def _cell_neighbors(self, cells: xp.ndarray):
        return xp.unique(self.face_face_connectivity[cells, :])

    def _find_neighbors(self, cell_line: xp.ndarray, connectivity: xp.ndarray) -> xp.ndarray:
        """Get a flattened list of all (unique) neighbors to a given global index list"""
        neighbors = connectivity[cell_line, :]
        shp = neighbors.shape
        unique_neighbors = xp.unique(neighbors.reshape(shp[0] * shp[1]))
        return unique_neighbors

    def find_edge_neighbors_for_cells(self, cell_line: xp.ndarray) -> xp.ndarray:
        return self._find_neighbors(cell_line, connectivity=self.face_edge_connectivity)



    def find_vertex_neighbors_for_cells(self, cell_line: xp.ndarray) -> xp.ndarray:
        return self._find_neighbors(cell_line, connectivity=self.face_node_connectivity)

    def owned_cells(self) -> xp.ndarray:
        """Returns the global indices of the cells owned by this rank"""
        owned_cells = self._mapping == self._props.rank
        return xp.asarray(owned_cells).nonzero()[0]

    # TODO (@halungge): move out of halo generator?
    def construct_decomposition_info(self) -> defs.DecompositionInfo:
        """
        Constructs the DecompositionInfo for the current rank.

        The DecompositionInfo object is constructed for all horizontal dimension starting from the
        cell distribution. Edges and vertices are then handled through their connectivity to the distributed cells.
        """

        #: cells
        owned_cells = self.owned_cells()  # global indices of owned cells
        first_halo_cells = self.next_halo_line(owned_cells)
        second_halo_cells = self.next_halo_line(first_halo_cells, owned_cells)

        total_halo_cells = xp.hstack((first_halo_cells, second_halo_cells))
        all_cells = xp.hstack((owned_cells, total_halo_cells))

        c_owner_mask = xp.isin(all_cells, owned_cells)

        decomp_info = defs.DecompositionInfo(klevels=self._num_levels).with_dimension(
            dims.CellDim, all_cells, c_owner_mask
        )

        #: edges
        edges_on_owned_cells = self.find_edge_neighbors_for_cells(owned_cells)
        edges_on_first_halo_line = self.find_edge_neighbors_for_cells(first_halo_cells)
        edges_on_second_halo_line = self.find_edge_neighbors_for_cells(second_halo_cells)

        all_edges = xp.hstack(
            (
                edges_on_owned_cells,
                xp.setdiff1d(edges_on_first_halo_line, edges_on_owned_cells),
                xp.setdiff1d(edges_on_second_halo_line, edges_on_first_halo_line),
            )
        )
        all_edges = xp.unique(all_edges)
        # We need to reduce the overlap:
        # `edges_on_owned_cells` and `edges_on_first_halo_line` both contain the edges on the cutting line.
        intersect_owned_first_line = xp.intersect1d(edges_on_owned_cells, edges_on_first_halo_line)

        def _update_owner_mask_by_max_rank_convention(
            owner_mask, all_indices, indices_on_cutting_line, target_connectivity
        ):
            """
            In order to have unique ownership of edges (and vertices) among nodes there needs to be
            a convention as to where those elements on the cutting line go:
            according to a remark in `mo_decomposition_tools.f90` ICON puts them to the node
            with the higher rank.

            # TODO (@halungge): can we add an assert for the target dimension of the connectivity being cells.
            Args:
                owner_mask: owner mask for the dimension
                all_indices: (global) indices of the dimension
                indices_on_cutting_line: global indices of the elements on the cutting line
                target_connectivity: connectivity matrix mapping the dimension d to faces
            Returns:
                updated owner mask
            """
            for index in indices_on_cutting_line:
                local_index = xp.nonzero(all_indices == index)[0][0]
                owning_ranks = self._mapping[target_connectivity[index]]
                assert (
                    xp.unique(owning_ranks).size > 1
                ), f"rank {self._props.rank}: all neighboring cells are owned by the same rank"
                assert (
                    self._props.rank in owning_ranks
                ), f"rank {self._props.rank}: neither of the neighboring cells: {owning_ranks} is owned by me"
                # assign the index to the rank with the higher rank
                if max(owning_ranks) > self._props.rank:
                    owner_mask[local_index] = False
                else:
                    owner_mask[local_index] = True
            return owner_mask

        # construct the owner mask
        edge_owner_mask = xp.isin(all_edges, edges_on_owned_cells)
        edge_owner_mask = _update_owner_mask_by_max_rank_convention(
            edge_owner_mask,
            all_edges,
            intersect_owned_first_line,
            self.edge_face_connectivity,
        )
        decomp_info.with_dimension(dims.EdgeDim, all_edges, edge_owner_mask)

        # vertices
        vertices_on_owned_cells = self.find_vertex_neighbors_for_cells(owned_cells)
        vertices_on_first_halo_line = self.find_vertex_neighbors_for_cells(first_halo_cells)
        vertices_on_second_halo_line = self.find_vertex_neighbors_for_cells(
            second_halo_cells
        )  # TODO (@halungge): do we need that?
        intersect_owned_first_line = xp.intersect1d(
            vertices_on_owned_cells, vertices_on_first_halo_line
        )

        # create decomposition_info for vertices
        all_vertices = xp.unique(xp.hstack((vertices_on_owned_cells, vertices_on_first_halo_line)))
        v_owner_mask = xp.isin(all_vertices, vertices_on_owned_cells)
        v_owner_mask = _update_owner_mask_by_max_rank_convention(
            v_owner_mask, all_vertices, intersect_owned_first_line, self.node_face_connectivity
        )
        decomp_info.with_dimension(dims.VertexDim, all_vertices, v_owner_mask)
        return decomp_info
    


    def construct_local_connectivity(self, 
        field_offset: gtx.FieldOffset, decomposition_info: defs.DecompositionInfo) -> xp.ndarray:
        """
        Construct a connectivity table for use on a given rank: it maps from source to target dimension in _local_ indices.

        Starting from the connectivity table on the global grid
        - we reduce it to the lines for the locally present entries of the the target dimension
        - the reduced connectivity then still maps to global source dimension indices:
            we replace those source dimension indices not present on the node to SKIP_VALUE and replace the rest with the local indices

        Args:
            field_offset: FieldOffset for which we want to construct the local connectivity table
            decomposition_info: DecompositionInfo for the current rank.

        Returns:
            connectivity are for the same FieldOffset but mapping from local target dimension indices to local source dimension indices.
        
        # TODO (@halungge): this should be done in the grid manager
        """
        source_dim = field_offset.source
        target_dim = field_offset.target[0]
        local_dim = field_offset.target[1]
        sliced_connectivity = self._connectivity(local_dim)[
            decomposition_info.global_index(target_dim, defs.DecompositionInfo.EntryType.ALL)
        ]
        log.debug(f"rank {self._props.rank} has local connectivity f: {sliced_connectivity}")

        global_idx = decomposition_info.global_index(
            source_dim, defs.DecompositionInfo.EntryType.ALL
        )

        # replace indices in the connectivity that do not exist on the local node by the SKIP_VALUE (those are for example neighbors of the outermost halo points)
        local_connectivity = xp.where(
            xp.isin(sliced_connectivity, global_idx), sliced_connectivity, SKIP_VALUE
        )

        # map to local source indices
        sorted_index_of_global_idx = xp.argsort(global_idx)
        global_idx_sorted = global_idx[sorted_index_of_global_idx]
        for i in xp.arange(local_connectivity.shape[0]):
            valid_neighbor_mask = local_connectivity[i, :] != SKIP_VALUE
            positions = xp.searchsorted(
                global_idx_sorted, local_connectivity[i, valid_neighbor_mask]
            )
            indices = sorted_index_of_global_idx[positions]
            local_connectivity[i, valid_neighbor_mask] = indices
        log.debug(f"rank {self._props.rank} has local connectivity f: {local_connectivity}")
        return local_connectivity


# should be done in grid manager!
    def local_grid(self,
                   props: defs.ProcessProperties,
                   decomposition_info: defs.DecompositionInfo,

                   limited_area: bool = False,
                   on_gpu: bool = False,
                   ) -> base_grid.BaseGrid:

        """
        Constructs a local grid for this rank based on the decomposition info.
        TODO (@halungge): for now only returning BaseGrid as we have not start/end indices implementation yet
        TODO (@halungge): make sure the INVALID_INDEX is set correctly: - when set in the original (global index) connectivity it should remain
        TODO (@halungge): how to handle the (source) indices of last halo line: their (global) neighbors are not all present on the local grid, set INVALID_INDEX (that is what xugrid does)
                                                                               check what ICON does, (they probably duplicate the valid indices...)
        Args:
            decomposition_info: the decomposition info for this rank
        Returns:
            local_grid: the local grid
        """
        num_vertices = decomposition_info.global_index(
            dims.VertexDim, defs.DecompositionInfo.EntryType.ALL
        ).size
        num_edges = decomposition_info.global_index(dims.EdgeDim,
                                                    defs.DecompositionInfo.EntryType.ALL).size
        num_cells = decomposition_info.global_index(dims.CellDim,
                                                    defs.DecompositionInfo.EntryType.ALL).size
        grid_size = base_grid.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        config = base_grid.GridConfig(
            horizontal_config=grid_size, vertical_size=self._num_levels, on_gpu=on_gpu, limited_area=limited_area
        )
    
        local_grid = (
            icon_grid.IconGrid(uuid.uuid4()).with_config(config).with_connectivities({
                dims.C2EDim: self.construct_local_connectivity(dims.C2EDim, decomposition_info),
                dims.E2CDim: self.construct_local_connectivity(dims.E2CDim, decomposition_info),
                dims.E2VDim: self.construct_local_connectivity(dims.E2VDim, decomposition_info),
                dims.V2EDim: self.construct_local_connectivity(dims.V2EDim, decomposition_info),
                dims.C2VDim: self.construct_local_connectivity(dims.C2VDim, decomposition_info),
                dims.V2CDim: self.construct_local_connectivity(dims.V2CDim, decomposition_info),
                dims.C2E2CDim: self.construct_local_connectivity(dims.C2E2CDim, decomposition_info),
                # dims.C2E2CODim
                # etc... # TODO (@halungge): this should all go to grid_manager
                    
                
                
            })
        )

    
    
        return local_grid


# TODO (@halungge): refine type hints: adjacency_matrix should be a connectivity matrix of C2E2C and
#  the return value an array of shape (n_cells,)


class Decomposer(Protocol):
    def __call__(self, adjacency_matrix, n_part: int) -> xp.ndarray:
        ...


class SimpleMetisDecomposer(Decomposer):
    """
    A simple decomposer using METIS for partitioning a grid topology.

    We use the simple pythonic interface to pymetis: just passing the adjacency matrix
    if more control is needed (for example by using weights we need to switch to the C like interface)
    https://documen.tician.de/pymetis/functionality.html
    """

    def __call__(self, adjacency_matrix, n_part: int) -> xp.ndarray:
        """
        Generate partition labesl for this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Args:
            n_part: int, number of partitions to create
        Returns: np.ndarray: array with partition label (int, rank number) for each cell
        """

        import pymetis

        cut_count, partition_index = pymetis.part_graph(nparts=n_part, adjacency=adjacency_matrix)
        return xp.array(partition_index)
