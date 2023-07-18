from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from typing_extensions import Self


class SupportsShapeAndSize(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    def __len__(self) -> int:
        ...


_ShapeProto_T = TypeVar("_ShapeProto_T", bound=SupportsShapeAndSize)


class GenericAdapter(Generic[_ShapeProto_T]):
    __slots__ = ("_data",)

    if TYPE_CHECKING:
        _data: Final[_ShapeProto_T]  # type: ignore
        _repr_meta_: str

    # - Constructors
    def _manager(self, data: _ShapeProto_T, inplace: bool) -> Self:
        if inplace:
            self._data = data  # type: ignore
            return self
        return self.__class__(data)

    def __init__(self, data: GenericAdapter | _ShapeProto_T) -> None:
        self._data = cast(_ShapeProto_T, data._data if isinstance(data, GenericAdapter) else data)

    # - Properties
    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def data(self) -> _ShapeProto_T:
        return self._data

    # - Dunder Methods
    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        repr_meta = getattr(self, "_repr_meta_", "...")
        return f"{self.__class__.__name__}[{repr_meta}] {repr(self._data)}"

    def _repr_html_(self) -> str:
        html = getattr(self._data, "_repr_html_", lambda: f"<pre>{repr(self._data)}</pre>")
        if not callable(html):
            raise TypeError(f"Expected {self._data!r} to have a _repr_html_ method")
        return html()


# =====================================================================================================================
# DataFrame Protocol Adapters
# =====================================================================================================================
_Column_co = TypeVar("_Column_co", covariant=True)
_Dtype_co = TypeVar("_Dtype_co", covariant=True)


class FrameProto(SupportsShapeAndSize, Generic[_Column_co, _Dtype_co], Protocol):
    @property
    def columns(self) -> _Column_co:
        ...

    @property
    def dtypes(self) -> _Dtype_co:
        ...


_SupportsColumnsAndDtypeT = TypeVar("_SupportsColumnsAndDtypeT", bound=FrameProto[Any, Any])


class GenericFrameAdapter(
    GenericAdapter[_SupportsColumnsAndDtypeT], Generic[_SupportsColumnsAndDtypeT, _Column_co, _Dtype_co]
):
    @property
    def columns(self) -> _Column_co:
        return self._data.columns

    @property
    def dtypes(self) -> _Dtype_co:
        return self._data.dtypes


# - polars.DataFrame

# these aliases are used to annotate DataFrame.__getitem__()
# MultiRowSelector indexes into the vertical axis and
# MultiColSelector indexes into the horizontal axis
# NOTE: wrapping these as strings is necessary for Python <3.10
MultiRowSelector: TypeAlias = "slice | range | list[int] | pl.Series"
MultiColSelector: TypeAlias = "slice | range | list[int] | list[str] | list[bool] | pl.Series"


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
    ) -> Self | pl.Series:
        if isinstance(item, (pl.Series, np.ndarray, pd.Series)) and item.dtype in (
            pl.Boolean,
            np.bool_,
            bool,
            pd.BooleanDtype,
        ):
            r = self._data.filter(item)
        else:
            r = self._data.__getitem__(item)  # type: ignore

        if isinstance(r, pl.DataFrame):
            return self._manager(r, inplace=False)
        return r


# - pandas.DataFrame
class PandasAdapter(GenericFrameAdapter[pd.DataFrame, pd.Index | pd.MultiIndex, pd.Series]):
    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self._data)

    def to_pandas(self) -> pd.DataFrame:
        return self._data

    def to_arrow(self) -> pa.Table:
        return pa.Table.from_pandas(self._data)
