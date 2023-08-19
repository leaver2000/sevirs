"""A mix of Datastructures and Adapters for working with Dataframe, Series, Arrays and Tensors."""
from __future__ import annotations

import abc
import logging
import random
import threading
from typing import (
    Any,
    Callable,
    Concatenate,
    Final,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    Protocol,
    Sized,
    TypeAlias,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import tqdm
from polars.type_aliases import IntoExpr
from typing_extensions import Self

from .._typing import FrameProtocol, Shaped

# Generator = tf.random.Generator
# =====================================================================================================================
# - Type Variables
# =====================================================================================================================
_P = ParamSpec("_P")
_T1 = TypeVar("_T1", bound=Any)
_T2 = TypeVar("_T2", bound=Any)
_T1_co = TypeVar("_T1_co", covariant=True)
_T2_co = TypeVar("_T2_co", covariant=True)
_ShapeProto_T = TypeVar("_ShapeProto_T", bound="Shaped")
_FrameProto_T = TypeVar("_FrameProto_T", bound="FrameProtocol[Any, Any]")


# =====================================================================================================================


class Indicies(Sized, Iterable[_T1_co], Protocol[_T1_co]):
    ...


class Dataset(Generic[_T1, _T2], abc.ABC):
    __slots__ = ("indices",)

    def __init__(
        self,
        indices: Indicies[_T1],
        **kwargs,
    ) -> None:
        super().__init__()
        self.indices: Final[tuple[_T1, ...]] = tuple(indices)
        self.n_proc: Final[int] = kwargs.pop("n_proc", 1)

    @abc.abstractmethod
    def get(self, index: _T1) -> _T2:
        ...
        
    def split(self, seed: int | None = None, frac: float = 0.8) -> tuple[Self, Self]:
        cls = type(self)
        if seed is not None:
            random.seed(seed)
        n = int(len(self.indices) * frac)
        indices = list(self.indices)
        random.shuffle(indices)
        left, right = indices[:n], indices[n:]
        return cls(left), cls(right)

    # - __dunder__
    def __getitem__(self, idx: int) -> _T2:
        return self.get(self.indices[idx])

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[_T2]:
        yield from (self.get(index) for index in self.indices)


class ContextualDataset(Dataset[_T1, _T2], abc.ABC):
    # =================================================================================================================
    # - abstractmethods
    @abc.abstractmethod
    def get_metadata(self, img_id: _T1 | None = None) -> pl.DataFrame:
        ...

    # =================================================================================================================
    # - overloads
    @overload  # type: ignore[misc]
    def select(self, index: _T1, *, metadata: Literal[False] = False) -> _T2:
        ...

    @overload
    def select(self, index: _T1, *, metadata: Literal[True] = True) -> tuple[_T2, pl.DataFrame]:
        ...

    def select(self, index: _T1, *, metadata: bool = False) -> _T2 | tuple[_T2, pl.DataFrame]:
        data = self.get(index)
        if metadata:
            return data, self.get_metadata(index)
        return data


class DataGenerator(ContextualDataset[_T1, _T2], Iterable[_T2], abc.ABC):
    # - overloads
    @overload  # type: ignore[misc]
    def iterate(self, *, metadata: Literal[False] = False) -> Iterable[_T2]:
        ...

    @overload
    def iterate(self, *, metadata: Literal[True] = True) -> Iterable[tuple[_T2, pl.DataFrame]]:
        ...

    def iterate(self, *, metadata: bool = False) -> Iterable[_T2 | tuple[_T2, pl.DataFrame]]:
        logging.info("🏃 Iterating over Dataset 🏃")
        for index in tqdm.tqdm(self.indices):
            yield self.select(index, metadata=metadata)  # type: ignore[call-overload]

    # =================================================================================================================




# =====================================================================================================================
class DataManager(Generic[_T1]):
    # - Constructors
    __slots__ = ("data",)

    def _manager(self, data: _T1 | Self, *, inplace: bool, **kwargs: Any) -> Self:
        if inplace:
            setattr(self, "data", data)
        else:
            self = self.__class__(data, **kwargs)
        return self

    def __init__(self, data: _T1 | Self) -> None:
        self.data: Final[_T1] = data.data if isinstance(data, DataManager) else data

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
# =====================================================================================================================
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
# =====================================================================================================================
class AbstractContextManager(abc.ABC):
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
        """Return `self` upon entering the runtime context."""
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @abc.abstractmethod
    def close(self) -> None:
        ...


class ThreadSafeIterator(Iterator[_T1]):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it: Generator[_T1, None, None], /) -> None:
        super().__init__()
        self.it: Final = it
        self.lock: Final = threading.Lock()

    def __iter__(self) -> ThreadSafeIterator[_T1]:
        return self

    def __next__(self) -> _T1:
        with self.lock:
            return next(self.it)
