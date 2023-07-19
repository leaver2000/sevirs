"""
A mix of abstract base classes and generic adapters for various data structures.
"""
from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Optional,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from typing_extensions import Self

# =====================================================================================================================
# - Type Variables
_T = TypeVar("_T", bound=Any)
_T1_co = TypeVar("_T1_co", covariant=True)
_T2_co = TypeVar("_T2_co", covariant=True)
_ShapeProto_T = TypeVar("_ShapeProto_T", bound="ShapeAndSizeProtocol")
_FrameProto_T = TypeVar("_FrameProto_T", bound="FrameProtocol[Any, Any]")


# =====================================================================================================================
# - Protocols
class ShapeAndSizeProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    def __len__(self) -> int:
        ...


class FrameProtocol(ShapeAndSizeProtocol, Generic[_T1_co, _T2_co], Protocol):
    @property
    def columns(self) -> _T1_co:
        ...

    @property
    def dtypes(self) -> _T2_co:
        ...


class SupportsClose(Protocol):
    def close(self) -> None:
        ...


# =====================================================================================================================
class GenericDataManager(Generic[_T]):
    __slots__ = ("_data",)

    if TYPE_CHECKING:
        _data: Final[_T]  # type: ignore[misc]

    # - Constructors
    def _manager(self, data: _T, inplace: bool) -> Self:
        if inplace:
            setattr(self, "_data", data)
            return self
        return self.__class__(data)

    def __init__(self, data: _T) -> None:
        self._data = data

    # - Properties
    @property
    def data(self) -> _T:
        return self._data


# =====================================================================================================================
# - Generic Adapters [Arrays, Series, Tensors, DataFrames, etc.]
class GenericDataAdapter(GenericDataManager[_ShapeProto_T]):
    if TYPE_CHECKING:
        _repr_meta_: Optional[Any]

    # - Properties
    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    # - Dunder Methods
    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        repr_meta = getattr(self, "_repr_meta_") or "..."
        return f"{self.__class__.__name__}[{repr_meta}] {repr(self._data)}"

    def _repr_html_(self) -> str:
        html = getattr(self._data, "_repr_html_", lambda: f"<pre>{repr(self._data)}</pre>")
        if not callable(html):
            raise TypeError(f"Expected {self._data!r} to have a _repr_html_ method")
        return html()


# =====================================================================================================================
# - DataFrames Adapters
class GenericFrameAdapter(GenericDataAdapter[_FrameProto_T], Generic[_FrameProto_T, _T1_co, _T2_co]):
    @property
    def columns(self) -> _T1_co:
        return self._data.columns

    @property
    def dtypes(self) -> _T2_co:
        return self._data.dtypes


# these aliases are used to annotate DataFrame.__getitem__()
# MultiRowSelector indexes into the vertical axis and
# MultiColSelector indexes into the horizontal axis
# NOTE: wrapping these as strings is necessary for Python <3.10
MultiRowSelector: TypeAlias = "slice | range | list[int] | pl.Series"
MultiColSelector: TypeAlias = "slice | range | list[int] | list[str] | list[bool] | pl.Series"


# - polars.DataFrame
class PolarsAdapter(GenericFrameAdapter[pl.DataFrame, list[str], list[pl.PolarsDataType]]):
    # - Transformations
    def to_polars(self) -> pl.DataFrame:
        return self._data

    def to_pandas(self) -> pd.DataFrame:
        return self._data.to_pandas()

    def to_arrow(self) -> pa.Table:
        return self._data.to_arrow()

    # - Selection Methods
    def select(self, *columns: str) -> Self:
        return self._manager(self._data.select(*columns), inplace=False)

    @overload
    def __getitem__(self, item: str) -> pl.Series:
        ...

    @overload
    def __getitem__(
        self,
        item: (
            int
            | np.ndarray[Any, Any]
            | MultiColSelector
            | tuple[int, MultiColSelector]
            | tuple[MultiRowSelector, MultiColSelector]
        ),
    ) -> Self:
        ...

    @overload
    def __getitem__(self, item: tuple[int, int | str]) -> Any:
        ...

    @overload
    def __getitem__(self, item: tuple[MultiRowSelector, int | str]) -> pl.Series:
        ...

    def __getitem__(
        self,
        item: (
            str
            | int
            | np.ndarray[Any, Any]
            | MultiColSelector
            | tuple[int, MultiColSelector]
            | tuple[MultiRowSelector, MultiColSelector]
            | tuple[MultiRowSelector, int | str]
            | tuple[int, int | str]
        ),
    ) -> Self | pl.Series | Any:
        if isinstance(item, (pl.Series, np.ndarray, pd.Series)) and item.dtype in (
            pd.BooleanDtype,
            pl.Boolean,
            np.bool_,
            bool,
        ):
            data = self._data.filter(item)
        else:
            data = self._data[item]  # type: ignore[assignment]

        if isinstance(data, pl.DataFrame):
            return self._manager(data, inplace=False)
        return data


# - pandas.DataFrame
class PandasAdapter(GenericFrameAdapter[pd.DataFrame, pd.Index | pd.MultiIndex, pd.Series]):
    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self._data)

    def to_pandas(self) -> pd.DataFrame:
        return self._data

    def to_arrow(self) -> pa.Table:
        return pa.Table.from_pandas(self._data)


# =====================================================================================================================
# - Abstract Classes
class AbstractDataCloser(GenericDataManager[_T], abc.ABC):
    """
    ```
    import io

    class Example(AbstractDataCloser[list[io.TextIOWrapper]]):
        def close(self) -> None:
            for x in self.data:
                x.close()
            self.data.clear()

    with Example([open(file) for file in ("file1.txt", "file2.txt")]) as x:
        print(len(x.data))
    print(len(x.data))
    ```
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @abc.abstractmethod
    def close(self) -> None:
        ...
