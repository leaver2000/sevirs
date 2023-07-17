from __future__ import annotations

__all__ = ["MultiIndexStore"]
import datetime
import typing

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas._libs.missing import NAType
from typing_extensions import Unpack

from ._typing import AnyT, Ts


class MultiIndexStore(typing.Mapping[Unpack[Ts], AnyT]):  # type: ignore
    """Allows for some of the multi index functionality of pandas Series objects but
    allows for data to be stored in a list. This is useful for storing data that is
    not easily stored in a pandas Series object either due to the data type or the
    or array interface.
    """

    __slots__ = ("_series", "_data", "_loc")
    if typing.TYPE_CHECKING:
        from pandas.core.series import _LocIndexerSeries

        _series: pd.Series[int]
        _loc: _LocIndexerSeries[int]
        _data: list[AnyT]

    # =================================================================================================================
    # Constructors
    def __init__(
        self,
        data: list[AnyT] | None = None,
        *,
        index: pd.MultiIndex | list[tuple[Unpack[Ts]]] | pd.DataFrame | NDArray,
        names: tuple[str, ...] | None = None,
    ) -> None:
        self._data = data or []

        # Convert the provided index to a MultiIndex
        if isinstance(index, list):
            index = pd.MultiIndex.from_tuples(index, names=names)
        elif isinstance(index, pd.DataFrame):
            index = pd.MultiIndex.from_frame(index, names=names)
        elif isinstance(index, np.ndarray):
            index = pd.MultiIndex.from_arrays(index.T, names=names)

        # there should not be any duplicates in the index
        if not index.nunique() == len(index):
            raise ValueError(f"Index {index} contains duplicates")
        self._series = s = pd.Series(
            range(len(data)) if data is not None else None, index=index, name=self.index_name, dtype="Int64"
        )
        self._loc = s.loc

    @classmethod
    def create(
        cls,
        index: typing.Mapping[typing.Any, typing.Sequence[tuple[Unpack[Ts]]]],
    ) -> MultiIndexStore[tuple[Unpack[Ts]], typing.Any]:
        return cls(index=list(index.values()), names=tuple(index.keys()))  # type: ignore

    # =================================================================================================================
    # Mapping interface
    @typing.overload
    def __getitem__(self, index: tuple[Unpack[Ts]]) -> AnyT:
        ...

    @typing.overload
    def __getitem__(
        self,
        index: typing.Union[
            slice,
            tuple[typing.Any, ...],  # single value index returns AnyT
            list[tuple[Unpack[Ts]]],
        ],
    ) -> list[AnyT]:
        ...

    def __getitem__(
        self,
        index: typing.Union[
            tuple[Unpack[Ts]],
            slice,
            tuple[typing.Any, ...],
            list[tuple[Unpack[Ts]]],
        ],
    ) -> AnyT | list[AnyT]:
        idx = typing.cast(
            "int | np.signedinteger[typing.Any] | pd.Series[int] | NAType",
            self._loc.__getitem__(index),  # type: ignore
        )
        if isinstance(idx, pd.Series) and not pd.isna(idx).any():
            return [self._data.__getitem__(i) for i in idx]
        elif isinstance(idx, (np.signedinteger, int)):
            return self._data.__getitem__(idx)  # type: ignore

        raise KeyError(
            f"Index {index} not found in {self.__class__.__name__} with index {self._series.index.tolist()}"
        )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> typing.Iterator[tuple[Unpack[Ts]]]:
        return iter(self._series.index)

    # =================================================================================================================
    # MutableMapping interface
    def __setitem__(self, key: tuple[Unpack[Ts]], data: AnyT) -> None:
        if key in self._series.index:
            self._data[self._loc[key]] = data  # type: ignore
        else:
            self._series[key] = len(self._data)
            self._data.append(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r}, index={self._series.index.tolist()!r})"

    # =================================================================================================================
    # Properties
    @property
    def index(self) -> pd.MultiIndex:
        return typing.cast(pd.MultiIndex, self._series.index)

    @property
    def index_name(self) -> str:
        return f"<{self.__class__.__name__}.index>"


def main() -> None:
    # =================================================================================================================

    store = MultiIndexStore[str, str, int, dict[str, str | None]](
        [
            {"hello": None},
            {"hello": "world2"},
        ],
        index=[
            ("a", "b", 1),
            ("a", "b", 2),
        ],
    )
    assert store[("a", "b", 1)]["hello"] is None
    store[("a", "b", 1)]["hello"] = "world"
    assert store[("a", "b", 1)]["hello"] == "world"
    # =================================================================================================================
    store = MultiIndexStore[int, int, str](["A", "B"], index=pd.DataFrame([[1, 1], [1, 2]]))
    assert store[1, 1] == "A"
    store[1, 2] = "C"
    assert store[1, 2] == "C"
    index = np.array(
        [
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
        ]
    ).astype(np.int_)
    # =================================================================================================================
    store = MultiIndexStore[int, int, int, str](["A", "B", "C", "D"], index=index)
    assert store[:, 1, :] == ["A", "B"]
    # =================================================================================================================
    store = MultiIndexStore.create({"foo": [(1, 1), (1, 2)], "bar": [(2, 1), (2, 2)]})
    assert store.index.names == ["foo", "bar"]
    # =================================================================================================================
    now = datetime.datetime.now()
    data = obj1, obj2 = [object(), object()]
    store = MultiIndexStore(
        data,
        index=[
            (now, 20, 20),
            (now + datetime.timedelta(hours=1), 20, 20),
        ],
    )
    assert store[now, 20, 20] is obj1 and store[now, 20, 20] is not obj2
    store[now, 20, 20] = obj2
    assert store[now, 20, 20] is obj2 and store[now, 20, 20] is not obj1


if __name__ == "__main__":
    main()
