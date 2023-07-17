# flake8: noqa
from __future__ import annotations

import sys
from typing import (
    Annotated,
    Any,
    Callable,
    Concatenate,
    NewType,
    ParamSpec,
    SupportsIndex,
    TypeAlias,
    TypeVar,
)

if sys.version_info < (3, 11):
    from typing_extensions import Self, TypeVarTuple, Unpack
else:
    from typing import Self, Unpack, TypeVarTuple

import typing

import numpy as np
import pandas as pd
from numpy.typing import NBitBase
from pandas._typing import HashableT, Scalar

# =====================================================================================================================
# TypeVariables and ParamSpecs
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
AnyT = TypeVar("AnyT", bound=typing.Any)
KeyT = TypeVar("KeyT", bound=typing.Hashable)
ValueT = TypeVar("ValueT")
ScalarT = TypeVar("ScalarT", bound=Scalar)
# =====================================================================================================================
# New Types
N = NewType(":", int)
N_T = TypeVar("N_T", bound=N)
Nd = Annotated[tuple[Unpack[Ts]], "number of dimensions"]
NdT = TypeVar("NdT", bound=typing.Sequence)

# =====================================================================================================================
if typing.TYPE_CHECKING:
    from pandas._typing import IndexType, MaskType
    from pandas.core.indexing import _IndexSliceTuple
else:
    _IndexSliceTuple = Any
    IndexType = Any
    MaskType = Any


ColumnIndexerType: TypeAlias = """(
slice
| HashableT
| IndexType
| MaskType
| typing.Callable[[pd.DataFrame], IndexType | MaskType | list[HashableT]]
| list[HashableT]
)"""
LocIndexerType: TypeAlias = """(
    int
    | ColumnIndexerType[HashableT]
    | tuple[
        IndexType | MaskType | list[HashableT] | slice | _IndexSliceTuple | Callable,
        list[HashableT] | slice | pd.Series[bool] | Callable
    ]
)"""
ImageIndexerType: TypeAlias = "slice | int | SupportsIndex"
# Concatenate[tuple[Unpack[Ts]], P],
from numpy.typing import NDArray

NumpyArray: TypeAlias = np.ndarray[Ts, np.dtype[AnyT]]  # type: ignore NDArray[tuple[Unpack[Ts]], P],


# np.dtype[AnyT],
