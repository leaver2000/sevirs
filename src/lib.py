from __future__ import annotations

__all__ = ["SEVIRBase", "html", "sel"]
import typing

import pandas as pd

try:
    import xarray as xr
except ImportError:
    xr = None
from numpy.typing import NDArray

from ._typing import LocIndexerType, Self

IndexT = typing.TypeVar("IndexT", pd.Index, pd.MultiIndex)
sel = pd.IndexSlice


def html(obj: typing.Any) -> str:
    if not hasattr(obj, "_repr_html_"):
        return repr(obj)
    return obj._repr_html_()


class SEVIRBase(typing.Generic[IndexT]):
    """A base class that is data agnostic and can be used to wrap any pandas DataFrame and support MultiIndex slicing."""

    __slots__ = ("_data",)

    def _manager(self, df: pd.DataFrame, inplace: bool) -> Self:
        if inplace:
            self._data = df
            return self
        return self.__class__(df)

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def __getitem__(self, idx: LocIndexerType[typing.Any]) -> pd.DataFrame:
        """I generic getitem method that returns the pandas DataFrame"""
        # because the index is a multi index the type checker does not understand that a single
        # integer will still return a pandas DataFrame and not a Series
        obj = self._data.loc.__getitem__(idx)
        return obj.to_frame() if isinstance(obj, pd.Series) else obj

    def __repr__(self) -> str:
        return repr(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _repr_html_(self) -> str:
        return html(self._data)

    @property
    def columns(self) -> pd.Index:
        return self._data.columns

    @property
    def index(self) -> IndexT:
        return self._data.index  # type: ignore

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def dtypes(self) -> pd.Series[str]:
        return self._data.dtypes

    def to_pandas(self, columns: list[str] | pd.Index | pd.MultiIndex | None = None) -> pd.DataFrame:
        """copy the data to a pandas DataFrame to that the original object is not modified."""
        df = self._data.copy()
        if columns is not None:
            df = df[columns]
        return df

    def to_xarray(self, columns) -> xr.Dataset:
        return xr.Dataset.from_dataframe(self.to_pandas(columns))

    def to_numpy(self, columns) -> NDArray:
        return self.to_pandas(columns).to_numpy()

    def get_level_values(self, level: str | int) -> pd.Index:
        return self._data.index.get_level_values(level)
