# flake8: noqa
from __future__ import annotations

__all__ = [
    "Self",
    # numpy types
    "NDArray",
    "DTypeLike",
    # pandas types
    "NAType",
    "NDArray",
    "ColumnIndexerType",
    "LocIndexerType",
    "IndexType",
    "MaskType",
]
import sys
import typing

AnyT = typing.TypeVar("AnyT", bound=typing.Any)
KeyT = typing.TypeVar("KeyT", bound=typing.Hashable)
ValueT = typing.TypeVar("ValueT")
if typing.TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        try:
            from typing_extensions import Self
        except ImportError:
            Self = typing.Any
    import pandas as pd
    from numpy.typing import DTypeLike, NDArray
    from pandas._libs.missing import NAType
    from pandas._typing import HashableT, IndexType, MaskType
    from pandas.core.indexing import _IndexSliceTuple

    ColumnIndexerType: typing.TypeAlias = """(
    slice
    | HashableT
    | IndexType
    | MaskType
    | typing.Callable[[pd.DataFrame], IndexType | MaskType | list[HashableT]]
    | list[HashableT]
    )"""
    LocIndexerType: typing.TypeAlias = """(
        int
        | ColumnIndexerType[HashableT]
        | tuple[
            IndexType | MaskType | list[HashableT] | slice | _IndexSliceTuple | typing.Callable,
            list[HashableT] | slice | pd.Series[bool] | typing.Callable
        ]
    )"""
    ImageIndexerType: typing.TypeAlias = "slice | int | typing.SupportsIndex"


else:
    Self = typing.Any
    # numpy types
    NDArray = typing.Any
    DTypeLike = typing.Any
    # pandas types
    NAType = typing.Any
    IndexType = typing.Any
    MaskType = typing.Any
    ColumnIndexerType = typing.Any
    LocIndexerType = typing.Any
    ImageIndexerType = typing.Any
