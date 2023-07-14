from __future__ import annotations

import typing
import abc
from numpy.typing import NDArray
import pandas as pd
import numpy as np

T = typing.TypeVar("T", bound=typing.Any)
KeyType = typing.TypeVar("KeyType", bound=typing.Hashable)
ValueType = typing.TypeVar("ValueType", bound=typing.Any)
ValueType_co = typing.TypeVar("ValueType_co", bound=typing.Any, covariant=True)
ValueType_contra = typing.TypeVar("ValueType_contra", bound=typing.Any, contravariant=True)
KeyStore: typing.TypeAlias = """(
    set[KeyType] 
    | list[KeyType]
    | NDArray
    | pd.Index
)"""


def is_key(key) -> typing.TypeGuard[KeyType]:
    return isinstance(key, typing.Hashable)


class MultiIndexMapping(typing.Mapping[KeyType, ValueType], abc.ABC):
    _data: typing.Mapping[KeyType, ValueType]

    def __init__(self, data: typing.Mapping[KeyType, ValueType]) -> None:
        self._data = data

    @typing.overload
    @abc.abstractmethod
    def __getitem__(self, __key: KeyType) -> ValueType:
        ...

    @typing.overload
    @abc.abstractmethod
    def __getitem__(self, __key: KeyStore[KeyType]) -> typing.Iterator[ValueType]:
        ...

    @abc.abstractmethod
    def __getitem__(self, __key: KeyType | KeyStore[KeyType]) -> typing.Iterator[ValueType] | ValueType:
        ...

    def to_dict(self, keys: KeyStore[KeyType] | None = None) -> dict[KeyType, ValueType]:
        return dict(zip(keys, self[keys]) if keys is not None else self._data)  # type: ignore

    def to_series(self, keys: KeyStore[KeyType] | None = None, name: str | None = None) -> pd.Series[ValueType]:
        return pd.Series(self.to_dict(keys), name=name)

    def to_frame(self, keys: KeyStore[KeyType] | None = None, index: list | None = None, dtype=None) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(keys), index=index, dtype=dtype)

    def to_numpy(self, keys: KeyStore[KeyType] | None = None) -> NDArray[ValueType]:
        if keys is None:
            return np.array(list(self._data.values()))
        return np.array(self[keys])


class StoreIterator(MultiIndexMapping[KeyType, ValueType]):
    def __iter__(self) -> typing.Iterator[KeyType]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, __key: KeyType) -> ValueType:
            ...

        @typing.overload
        def __getitem__(self, __key: KeyStore[KeyType]) -> typing.Iterator[ValueType]:
            ...

    def __getitem__(self, __key: KeyType | KeyStore[KeyType]) -> typing.Iterator[ValueType] | ValueType:
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

    x = s["A"]
    size = 90 * 49 * 49 * 49
    s = StoreIterator(
        {
            "vis": np.arange(size).reshape(90, 49, 49, 49),
            "vil": np.arange(size).reshape(90, 49, 49, 49),
            "ir_069": np.arange(size).reshape(90, 49, 49, 49),
            "ir_109": np.arange(size).reshape(90, 49, 49, 49),
        }
    )
    print(np.array([a[1:2, :, :, 1:2] for a in s[["vis", "vil"]]]))


if __name__ == "__main__":
    main()
