"""A mix of Abstract Base Classes and Generic Data Adapters for various data structures."""
from __future__ import annotations

import abc
import contextlib
import dataclasses
from typing import (
    Any,
    Callable,
    Concatenate,
    Final,
    Generic,
    Iterable,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from polars.type_aliases import IntoExpr
from typing_extensions import Self

from .constants import (
    EVENT_TYPE,
    FILE_INDEX,
    FILE_NAME,
    FILE_REF,
    ID,
    IMG_TYPE,
    TIME_UTC,
    ImageType,
)

# =====================================================================================================================
# - Type Variables
# =====================================================================================================================
_P = ParamSpec("_P")
_T1 = TypeVar("_T1", bound=Any)
_T2 = TypeVar("_T2", bound=Any)
_T1_co = TypeVar("_T1_co", covariant=True)
_T2_co = TypeVar("_T2_co", covariant=True)
_ShapeProto_T = TypeVar("_ShapeProto_T", bound="ShapeAndSizeProtocol")
_FrameProto_T = TypeVar("_FrameProto_T", bound="FrameProtocol[Any, Any]")


# =====================================================================================================================
# - Protocols
# =====================================================================================================================
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
# - Generics and Base Classes
# =====================================================================================================================
@dataclasses.dataclass
class BaseConfig:
    inputs: tuple[ImageType, ...]
    targets: tuple[ImageType, ...]

    def __post_init__(self) -> None:
        self.inputs = tuple(self.inputs)
        self.targets = tuple(self.targets)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class DataManager(Generic[_T1]):
    __slots__ = ("_data",)

    # - Constructors
    def _manager(self, data: _T1 | Self, *, inplace: bool, **kwargs: Any) -> Self:
        if inplace:
            setattr(self, "_data", data)
        else:
            self = self.__class__(data, **kwargs)
        return self

    def __init__(self, data: _T1 | Self) -> None:
        self._data: Final[_T1] = data.data if isinstance(data, DataManager) else data

    @property
    def data(self) -> _T1:
        return self._data

    def pipe(self, __func: Callable[Concatenate[(Self, _P)], _T2], /, *args: _P.args, **kwargs: _P.kwargs) -> _T2:
        """
        ```python
        m = GenericDataManager({"s": 1})

        def func(mgr: GenericDataManager[dict[str, int]], x: int) -> int:
            return mgr.data["s"] + x

        n = m.pipe(func, 1) # type: int
        ```
        """
        return __func(self, *args, **kwargs)

    def __call__(
        self, __func: Callable[Concatenate[(Self, _P)], _T1 | Self], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Self:
        """
        Parameters
        ----------
        __func : Callable[Concatenate[(Self, _P)], _T1] a callable that takes an instance of a DataManager as its first
        argument and returns either the underlying data or a new instance of a DataManager


        ```python
        def my_filter(self: sevir.Catalog) -> pl.DataFrame:
            mask = self.id == "R18032505027684"
            return self.data.filter(mask)

        assert isinstance(catalog, sevir.Catalog)
        ```
        """

        data = self.pipe(__func, *args, **kwargs)
        return self._manager(data, inplace=False)


# - Generic Adapters [NDArrays, Series, Tensors, DataFrames, etc.]
class DataAdapter(DataManager[_ShapeProto_T]):
    # - Properties
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    # - Dunder Methods
    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"[{self.__class__.__name__}]\n{repr(self.data)}"

    def _repr_html_(self) -> str:
        html = getattr(self.data, "_repr_html_", lambda: f"<pre>{repr(self.data)}</pre>")
        if not callable(html):
            raise TypeError(f"Expected {self.data!r} to have a _repr_html_ method")
        return html()


# =====================================================================================================================
# - DataFrames Adapters
class FrameAdapter(DataAdapter[_FrameProto_T], Generic[_FrameProto_T, _T1_co, _T2_co]):
    @property
    def columns(self) -> _T1_co:
        return self.data.columns

    @property
    def dtypes(self) -> _T2_co:
        return self.data.dtypes


# these aliases are used to annotate DataFrame.__getitem__()
# MultiRowSelector indexes into the vertical axis and
# MultiColSelector indexes into the horizontal axis
MultiRowSelector: TypeAlias = "slice | range | list[int] | pl.Series"
MultiColSelector: TypeAlias = "slice | range | list[int] | list[str] | list[bool] | pl.Series"


# - polars.DataFrame
class PolarsAdapter(FrameAdapter[pl.DataFrame, list[str], list[pl.PolarsDataType]]):
    # - Selection Methods
    def select(self, *columns: str) -> Self:
        return self._manager(self.data.select(*columns), inplace=False)

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], inplace: bool = False, **named_exprs: IntoExpr
    ) -> Self:
        return self._manager(self.data.with_columns(*exprs, **named_exprs), inplace=inplace)

    # - Transformations
    def to_polars(self) -> pl.DataFrame:
        return self.data

    def to_pandas(self) -> pd.DataFrame:
        return self.data.to_pandas()

    def to_arrow(self) -> pa.Table:
        return self.data.to_arrow()

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
            data = self.data.filter(item)
        else:
            data = self.data[item]  # type: ignore[assignment]

        if isinstance(data, pl.DataFrame):
            return self._manager(data, inplace=False)
        return data


# - pandas.DataFrame
class PandasAdapter(FrameAdapter[pd.DataFrame, pd.Index | pd.MultiIndex, pd.Series]):
    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self.data)

    def to_pandas(self) -> pd.DataFrame:
        return self.data

    def to_arrow(self) -> pa.Table:
        return pa.Table.from_pandas(self.data)


# =====================================================================================================================
# - Abstract Classes
class AbstractContextManager(contextlib.AbstractContextManager[_T1]):
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

    def __exit__(self, *_) -> None:
        self.close()

    @abc.abstractmethod
    def close(self) -> None:
        ...


class AbstractCatalog(abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> pl.DataFrame:
        ...

    @property
    def columns(self) -> list[str]:
        return self.data.columns

    @property
    def id(self) -> pl.Series:  # noqa: A003
        return self.data[ID]

    @property
    def file_name(self) -> pl.Series:
        return self.data[FILE_NAME]

    @property
    def file_index(self) -> pl.Series:
        return self.data[FILE_INDEX]

    @property
    def file_ref(self) -> pl.Series:
        return self.data[FILE_REF]

    @property
    def img_type(self) -> pl.Series:
        return self.data[IMG_TYPE]

    @property
    def event_type(self) -> pl.Series:
        return self.data[EVENT_TYPE]

    @property
    def time_utc(self) -> pl.Series:
        return self.data[TIME_UTC]
