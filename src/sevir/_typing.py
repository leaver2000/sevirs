# flake8: noqa
from __future__ import annotations

import enum
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    get_args,
)

import numpy as np
import pandas as pd
from pandas._typing import HashableT, Scalar

if sys.version_info < (3, 11):
    from typing_extensions import TypeVarTuple, Unpack
else:
    from typing import Self, TypeVarTuple, Unpack

if TYPE_CHECKING:
    from pandas._typing import IndexType, MaskType
    from pandas.core.indexing import _IndexSliceTuple
else:
    _IndexSliceTuple = Any
    IndexType = Any
    MaskType = Any

# =====================================================================================================================
Ts = TypeVarTuple("Ts")
AnyT = TypeVar("AnyT", bound=Any)
KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
ScalarT = TypeVar("ScalarT", bound=Scalar)
DictStr = dict[str, AnyT]
DictStrAny = DictStr[Any]


def cast_literal_list(cls: type[ValueT]) -> ValueT:
    """
    >>> Numbers = typing.Literal[1, 2, 3]
    >>> NUM_LIST = ONE, TWO, THREE = cast_literal_list(list[Numbers])
    >>> NUM_LIST
    [1, 2, 3]
    """
    return list(get_args(get_args(cls)[0]))  # type: ignore[return-value]


# =====================================================================================================================
class Nd(Generic[Unpack[Ts]]):
    """type alias for a tuple of ints or slices
    >>> import numpy as np
    >>> from sevir._typing import Nd
    >>> a: np.ndarray[Nd[2, 2], np.int64] = np.array([[1, 2], [3, 4]])
    """


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
ColumnIndexerType: TypeAlias = """(
slice
| HashableT
| IndexType
| MaskType
| Callable[[pd.DataFrame], IndexType | MaskType | list[HashableT]]
| list[HashableT]
)"""
LocIndexerType: TypeAlias = """(
    int
    | ColumnIndexerType
    | tuple[
        IndexType | MaskType | list[HashableT] | slice | _IndexSliceTuple | Callable,
        list[HashableT] | slice | pd.Series[bool] | Callable
    ]
)"""
ImageIndexerType: TypeAlias = "slice | int | SupportsIndex"
