from __future__ import annotations

import abc
import typing

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ._typing import AnyT, KeyT, ValueT


class MappedDataset(Dataset[ValueT], typing.Generic[KeyT, ValueT], abc.ABC):
    ...


KeyStore: typing.TypeAlias = """(
    set[KeyT] 
    | list[KeyT]
    | NDArray
    | pd.Index
)"""


def is_key(key: KeyT | typing.Any) -> typing.TypeGuard[KeyT]:
    return isinstance(key, typing.Hashable)


class MultiIndexMapping(typing.Mapping[KeyT, AnyT], abc.ABC):
    _data: typing.Mapping[KeyT, AnyT]

    def __init__(self, data: typing.Mapping[KeyT, AnyT]) -> None:
        self._data = data
        self._index = np.arange(len(data))

    @property
    def data(self) -> typing.Mapping[KeyT, AnyT]:
        import copy

        return copy.deepcopy(self._data)

    @property
    def index(self) -> NDArray[np.int_]:
        return self._index

    if typing.TYPE_CHECKING:

        @abc.abstractmethod
        @typing.overload
        def __getitem__(self, __key: KeyT) -> AnyT:
            ...

        @abc.abstractmethod
        @typing.overload
        def __getitem__(self, __key: typing.Sequence[KeyT]) -> typing.Sequence[AnyT]:
            ...

    @abc.abstractmethod
    def __getitem__(self, __key: KeyT | typing.Sequence[KeyT]) -> typing.Sequence[ValueT] | ValueT:
        ...

    def to_dict(self, keys: typing.Sequence[KeyT] | None = None) -> dict[KeyT, AnyT]:
        return dict(zip(keys, self[keys]) if keys is not None else self._data)  # type: ignore

    def to_series(self, keys: typing.Sequence[KeyT] | None = None, name: str | None = None) -> pd.Series[AnyT]:
        return pd.Series(self.to_dict(keys), name=name)

    def to_frame(
        self, keys: typing.Sequence[KeyT] | None = None, index: list | None = None, dtype=None
    ) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(keys), index=index, dtype=dtype)

    def to_numpy(self, keys: typing.Sequence[KeyT] | None = None) -> NDArray[AnyT]:
        if keys is None:
            return np.array(list(self._data.values()))
        return np.array(self[keys])


class StoreIterator(MultiIndexMapping[KeyT, ValueT]):
    def __iter__(self) -> typing.Iterator[KeyT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, __key: KeyT) -> ValueT:
            ...

        @typing.overload
        def __getitem__(self, __key: KeyStore[KeyT]) -> typing.Iterator[ValueT]:
            ...

    def __getitem__(self, __key: KeyT | KeyStore[KeyT]) -> typing.Iterator[ValueT] | ValueT:
        if is_key(__key):
            return self._data[__key]
        if isinstance(__key, typing.Iterable):
            return iter(self._data[k] for k in __key)

        raise KeyError(__key)


def main():
    assert StoreIterator({"A": "1"}).to_dict() == {"A": "1"}  # StoreIterator[str, int]
    assert StoreIterator({"A": ["1"]}).to_dict() == {"A": ["1"]}  # StoreIterator[str, list[int]]
    s = StoreIterator({"A": ["1"], "B": ["1"], "C": ["1"]})
    assert s.to_dict(["A", "B"]) == {"A": ["1"], "B": ["1"]}
    assert np.all(s.to_numpy() == np.array([["1"], ["1"], ["1"]]))

    size = 90 * 49 * 49 * 49
    s = StoreIterator(
        {
            "vis": np.arange(size).reshape(90, 49, 49, 49),
            "vil": np.arange(size).reshape(90, 49, 49, 49),
            "ir_069": np.arange(size).reshape(90, 49, 49, 49),
            "ir_109": np.arange(size).reshape(90, 49, 49, 49),
        }
    )
    s["vis"]
    print(np.array([a[1:2, :, :, 1:2] for a in s[["vis", "vil"]]]))


if __name__ == "__main__":
    main()
