# flake8: noqa
from __future__ import annotations

import sys
from typing import (
    Annotated,
    Any,
    Callable,
    NewType,
    ParamSpec,
    Sequence,
    SupportsIndex,
    TypeAlias,
    TypeVar,
)
import enum

if sys.version_info < (3, 11):
    from typing_extensions import Self, TypeVarTuple, Unpack
else:
    from typing import Self, Unpack, TypeVarTuple

import typing

import numpy as np
import pandas as pd
from pandas._typing import HashableT, Scalar

# =====================================================================================================================
Ts = TypeVarTuple("Ts")
AnyT = TypeVar("AnyT", bound=typing.Any)
KeyT = TypeVar("KeyT", bound=typing.Hashable)
ValueT = TypeVar("ValueT")
ScalarT = TypeVar("ScalarT", bound=Scalar)


# =====================================================================================================================
class Nd(tuple[Unpack[Ts]]):
    """type alias for a tuple of ints or slices
    >>> import numpy as np
    >>> from sevir._typing import Nd
    >>> a: np.ndarray[Nd[2, 2], np.int64] = np.array([[1, 2], [3, 4]])
    """

    # def


N = enum.Enum(":", {"_": slice(None)})
_NdT = TypeVar("_NdT", bound=Nd, contravariant=True)
Array: TypeAlias = np.ndarray[_NdT, np.dtype[AnyT]]
"""
>>> from typing_extensions import reveal_type
>>> import numpy as np
>>> from sevir._typing import Array, Nd, N
>>> a: Array[Nd[N, N], np.int64] = np.array([[1, 2], [3, 4]])
>>> reveal_type(a)
Runtime type is 'ndarray'
"""


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
