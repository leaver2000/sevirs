from __future__ import annotations

import typing
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


class StoreIterator(typing.Mapping[KeyType, ValueType]):
    @typing.overload
    def __init__(
        self,
        data: typing.Mapping[KeyType, ValueType],
        in_to: None = ...,
    ) -> None:
        ...

    @typing.overload
    def __init__(
        self,
        data: typing.Mapping[KeyType, typing.Any],
        in_to: typing.Callable[..., ValueType] = ...,  # type: ignore[assignment]
    ) -> None:
        ...

    def __init__(
        self,
        data: typing.Mapping[KeyType, ValueType | typing.Any],
        in_to: typing.Callable[..., ValueType] | None = None,
    ) -> None:
        # print(typing.get_type_hints(in_to), in_to, typing.get_args(in_to))

        # def resolve_into():
        #     args = typing.get_args(in_to)
        #     if not args:
        #         return in_to

        #     def inner(arg, value: typing.Any) -> ValueType:
        #         return arg(value)
        #     return

        if data is not None:
            data = {k: in_to(v) for k, v in data.items()}
        self._data = data

    @typing.overload
    def __getitem__(self, __key: KeyType) -> ValueType:
        ...

    @typing.overload
    def __getitem__(self, __key: KeyStore[KeyType]) -> typing.Iterator[ValueType]:
        ...

    def __getitem__(self, __key: KeyType | KeyStore[KeyType]) -> typing.Iterator[ValueType] | ValueType:
        if isinstance(__key, typing.Hashable):
            return self._data[__key]
        return iter(self._data[k] for k in __key)  # type: ignore[return-value]

    def __iter__(self) -> typing.Iterator[KeyType]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self, keys: KeyStore[KeyType] | None = None) -> dict[KeyType, ValueType]:
        return dict(self[keys] if keys else self._data)

    def to_series(self, keys: KeyStore[KeyType] | None = None, name: str | None = None) -> pd.Series[ValueType]:
        return pd.Series(self.to_dict(keys), name=name)

    def to_frame(self, keys: KeyStore[KeyType] | None = None, index: list | None = None, dtype=None) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(keys), index=index, dtype=dtype)

    def to_numpy(self, keys: KeyStore[KeyType] | None = None, dtype=None) -> NDArray[ValueType]:
        if keys is None:
            return np.array(list(self._data.values()))
        return np.array([self[k] for k in keys])


def main():
    assert StoreIterator({"A": "1"}, in_to=int).to_dict() == {"A": 1}  # StoreIterator[str, int]
    StoreIterator({"A": ["1"]}, in_to=list[int]).to_dict() == {"A": [1]}  # StoreIterator[str, list[int]]


if __name__ == "__main__":
    main()
