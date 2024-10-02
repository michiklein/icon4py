# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import enum
import functools
import inspect
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    get_args,
)

import gt4py.next as gtx
import gt4py.next.ffront.decorator as gtx_decorator
import xarray as xa

from icon4py.model.common import dimension as dims, exceptions, settings
from icon4py.model.common.grid import (
    base as base_grid,
    horizontal as h_grid,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.settings import xp
from icon4py.model.common.states import metadata as metadata, model, utils as state_utils
from icon4py.model.common.utils import builder


DomainType = TypeVar("DomainType", h_grid.Domain, v_grid.Domain)


class RetrievalType(enum.Enum):
    FIELD = 0
    DATA_ARRAY = 1
    METADATA = 2


def check_setup(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_setup():
            raise exceptions.IncompleteSetupError(
                "Factory not fully instantiated, missing grid or allocator"
            )
        return func(self, *args, **kwargs)

    return wrapper


class FieldProvider(Protocol):
    """
    Protocol for field providers.

    A field provider is responsible for the computation and caching of a set of fields.
    The fields can be accessed by their field_name (str).

    A FieldProvider is a callable that has three methods (except for __call__):
     - evaluate (abstract) : computes the fields based on the instructions of the concrete implementation
     - fields: Mapping of a field_name to list of field names provided by the provider
     - dependencies: returns a list of field_names that the fields provided by this provider depend on.

    """
    

    @abc.abstractmethod
    def evaluate(self, factory: "FieldsFactory") -> None:
        ...

    def __call__(self, field_name: str, factory: "FieldsFactory") -> state_utils.FieldType:
        if field_name not in self.fields:
            raise ValueError(f"Field {field_name} not provided by f{self.func.__name__}.")
        if any([f is None for f in self.fields.values()]):
            self.evaluate(factory)
        return self.fields[field_name]

    @property
    def dependencies(self) -> Sequence[str]:
        ...

    @property
    def fields(self) -> Mapping[str, Any]:
        ...
    
    @property
    def func(self)->Callable:
        ...

class PrecomputedFieldProvider(FieldProvider):
    """Simple FieldProvider that does not do any computation but gets its fields at construction and returns it upon provider.get(field_name)."""

    def __init__(self, fields: dict[str, state_utils.FieldType]):
        self._fields = fields

    def evaluate(self, factory: "FieldsFactory") -> None:
        pass

    @property
    def dependencies(self) -> Sequence[str]:
        return []

    def __call__(self, field_name: str, factory: "FieldsFactory") -> state_utils.FieldType:
        return self.fields[field_name]
    
    # TODO signature should this only return the field_names produced by this provider?
    @property
    def fields(self) -> Mapping[str, Any]:
        return self._fields
    
 
    @property
    def func(self) -> Callable:
        return lambda : self.fields 


class ProgramFieldProvider(FieldProvider):
    """
    Computes a field defined by a GT4Py Program.

    Args:
        func: GT4Py Program that computes the fields
        domain: the compute domain used for the stencil computation
        fields: dict[str, str], fields computed by this stencil:  the key is the variable name of the out arguments used in the program and the value the name the field is registered under and declared in the metadata.
        deps: dict[str, str], input fields used for computing this stencil: the key is the variable name used in the program and the value the name of the field it depends on.
        params: scalar parameters used in the program
    """

    def __init__(
        self,
        func: gtx_decorator.Program,
        domain: dict[gtx.Dimension, tuple[DomainType, DomainType]],
        fields: dict[str, str],
        deps: dict[str, str],
        params: Optional[dict[str, state_utils.Scalar]] = None,
    ):
        self._func = func
        self._compute_domain = domain
        self._dependencies = deps
        self._output = fields
        self._params = params if params is not None else {}
        self._fields: dict[str, Optional[gtx.Field | state_utils.Scalar]] = {
            name: None for name in fields.values()
        }

    def _unallocated(self) -> bool:
        return not all(self._fields.values())

    def _allocate(self, allocator, grid: base_grid.BaseGrid) -> dict[str, state_utils.FieldType]:
        def _map_size(dim: gtx.Dimension, grid: base_grid.BaseGrid) -> int:
            if dim == dims.KHalfDim:
                return grid.num_levels + 1
            return grid.size[dim]

        def _map_dim(dim: gtx.Dimension) -> gtx.Dimension:
            if dim == dims.KHalfDim:
                return dims.KDim
            return dim

        field_domain = {
            _map_dim(dim): (0, _map_size(dim, grid)) for dim in self._compute_domain.keys()
        }
        return {
            k: allocator(field_domain, dtype=metadata.attrs[k]["dtype"])
            for k in self._fields.keys()
        }

    # TODO (@halungge) this can be simplified when completely disentangling vertical and horizontal grid.
    #   the IconGrid should then only contain horizontal connectivities and no longer any Koff which should be moved to the VerticalGrid
    def _get_offset_providers(
        self, grid: icon_grid.IconGrid, vertical_grid: v_grid.VerticalGrid
    ) -> dict[str, gtx.FieldOffset]:
        offset_providers = {}
        for dim in self._compute_domain.keys():
            if dim.kind == gtx.DimensionKind.HORIZONTAL:
                horizontal_offsets = {
                    k: v
                    for k, v in grid.offset_providers.items()
                    if isinstance(v, gtx.NeighborTableOffsetProvider)
                    and v.origin_axis.kind == gtx.DimensionKind.HORIZONTAL
                }
                offset_providers.update(horizontal_offsets)
            if dim.kind == gtx.DimensionKind.VERTICAL:
                vertical_offsets = {
                    k: v
                    for k, v in grid.offset_providers.items()
                    if isinstance(v, gtx.Dimension) and v.kind == gtx.DimensionKind.VERTICAL
                }
                offset_providers.update(vertical_offsets)
        return offset_providers

    def _domain_args(
        self, grid: icon_grid.IconGrid, vertical_grid: v_grid.VerticalGrid
    ) -> dict[str : gtx.int32]:
        domain_args = {}

        for dim in self._compute_domain:
            if dim.kind == gtx.DimensionKind.HORIZONTAL:
                domain_args.update(
                    {
                        "horizontal_start": grid.start_index(self._compute_domain[dim][0]),
                        "horizontal_end": grid.end_index(self._compute_domain[dim][1]),
                    }
                )
            elif dim.kind == gtx.DimensionKind.VERTICAL:
                domain_args.update(
                    {
                        "vertical_start": vertical_grid.index(self._compute_domain[dim][0]),
                        "vertical_end": vertical_grid.index(self._compute_domain[dim][1]),
                    }
                )
            else:
                raise ValueError(f"DimensionKind '{dim.kind}' not supported in Program Domain")
        return domain_args

    def evaluate(self, factory: "FieldsFactory"):
        self._fields = self._allocate(factory.allocator, factory.grid)
        deps = {k: factory.get(v) for k, v in self._dependencies.items()}
        deps.update(self._params)
        deps.update({k: self._fields[v] for k, v in self._output.items()})
        dims = self._domain_args(factory.grid, factory.vertical_grid)
        offset_providers = self._get_offset_providers(factory.grid, factory.vertical_grid)
        deps.update(dims)
        self._func.with_backend(factory._backend)(**deps, offset_provider=offset_providers)

    @property
    def fields(self) -> Mapping[str, Any]:
        return self._fields
    
    @property
    def func(self) ->Callable:
        return self._func
    @property
    def dependencies(self) -> Sequence[str]:
        return list(self._dependencies.values())
        

class NumpyFieldProvider(FieldProvider):
    """
    Computes a field defined by a numpy function.

    Args:
        func: numpy function that computes the fields
        domain: the compute domain used for the stencil computation
        fields: Seq[str] names under which the results fo the function will be registered
        deps: dict[str, str] input fields used for computing this stencil: the key is the variable name used in the program and the value the name of the field it depends on.
        params: scalar arguments for the function
    """

    def __init__(
        self,
        func: Callable,
        domain: dict[gtx.Dimension : tuple[DomainType, DomainType]],
        fields: Sequence[str],
        deps: dict[str, str],
        offsets: Optional[dict[str, gtx.Dimension]] = None,
        params: Optional[dict[str, state_utils.Scalar]] = None,
    ):
        self._func = func
        self._compute_domain = domain
        self._offsets = offsets
        self._dims = domain.keys()
        self._fields: dict[str, Optional[state_utils.FieldType]] = {name: None for name in fields}
        self._dependencies = deps
        self._offsets = offsets if offsets is not None else {}
        self._params = params if params is not None else {}

    def evaluate(self, factory: "FieldsFactory") -> None:
        self._validate_dependencies()
        args = {k: factory.get(v).ndarray for k, v in self._dependencies.items()}
        offsets = {k: factory.grid.connectivities[v] for k, v in self._offsets.items()}
        args.update(offsets)
        args.update(self._params)
        results = self._func(**args)
        ## TODO: can the order of return values be checked?
        results = (results,) if isinstance(results, xp.ndarray) else results

        self._fields = {
            k: gtx.as_field(tuple(self._dims), results[i]) for i, k in enumerate(self.fields)
        }

    def _validate_dependencies(self):
        func_signature = inspect.signature(self._func)
        parameters = func_signature.parameters
        for dep_key in self._dependencies.keys():
            parameter_definition = parameters.get(dep_key)
            assert parameter_definition.annotation == xp.ndarray, (
                f"Dependency {dep_key} in function {self._func.__name__}:  does not exist or has "
                f"or has wrong type ('expected np.ndarray') in {func_signature}."
            )

        for param_key, param_value in self._params.items():
            parameter_definition = parameters.get(param_key)
            checked = _check(
                parameter_definition, param_value, union=state_utils.IntegerType
            ) or _check(parameter_definition, param_value, union=state_utils.FloatType)
            assert checked, (
                f"Parameter {param_key} in function {self._func.__name__} does not "
                f"exist or has the wrong type: {type(param_value)}."
            )

    @property
    def func(self) ->Callable:
        return self._func
    
    @property
    def dependencies(self) -> Sequence[str]:
        return list(self._dependencies.values())
    
    @property
    def fields(self) -> Mapping[str, Any]:
        return self._fields

def _check(
    parameter_definition: inspect.Parameter,
    value: Union[state_utils.Scalar, gtx.Field],
    union: Union,
) -> bool:
    members = get_args(union)
    return (
        parameter_definition is not None
        and parameter_definition.annotation in members
        and type(value) in members
    )


class FieldsFactory:
    def __init__(
        self,
        metadata: dict[str, model.FieldMetaData],
        grid: icon_grid.IconGrid = None,
        vertical_grid: v_grid.VerticalGrid = None,
        backend=None,
        
        
    ):
        self._metadata = metadata
        self._grid = grid
        self._vertical = vertical_grid
        self._providers: dict[str, "FieldProvider"] = {}
        self._backend = backend
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)

    """
    Factory for fields.

    Lazily compute fields and cache them.
    """

    def is_setup(self):
        return self._grid is not None and self.backend is not None

    @builder.builder
    def with_grid(self, grid: base_grid.BaseGrid, vertical_grid: v_grid.VerticalGrid):
        self._grid = grid
        self._vertical = vertical_grid

    @builder.builder
    def with_backend(self, backend=settings.backend):
        self._backend = backend
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)

    @property
    def backend(self):
        return self._backend

    @property
    def grid(self):
        return self._grid

    @property
    def vertical_grid(self):
        return self._vertical

    @property
    def allocator(self):
        return self._allocator

    def register_provider(self, provider: FieldProvider):
        for dependency in provider.dependencies:
            if dependency not in self._providers.keys():
                raise ValueError(f"Dependency '{dependency}' not found in registered providers")

        for field in provider.fields:
            self._providers[field] = provider

    @check_setup
    def get(
        self, field_name: str, type_: RetrievalType = RetrievalType.FIELD
    ) -> Union[state_utils.FieldType, xa.DataArray, model.FieldMetaData]:
        if field_name not in self._providers:
            raise ValueError(f"Field {field_name} not provided by the factory")
        if type_ == RetrievalType.METADATA:
            return self._metadata[field_name]
        if type_ in (RetrievalType.FIELD,RetrievalType.DATA_ARRAY):
            provider = self._providers[field_name]
            buffer = provider(field_name, self)
            return buffer if type_ == RetrievalType.FIELD else state_utils.to_data_array(buffer, self._metadata[field_name])
            
       
        raise ValueError(f"Invalid retrieval type {type_}")
