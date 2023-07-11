from __future__ import annotations

import typing

from typing_extensions import Self

K = typing.TypeVar("K", bound=typing.Hashable)
V = typing.TypeVar("V")


class Mapping(typing.Mapping[K, V]):
    __slots__ = ("_data",)

    def __init__(self, data: typing.Mapping[K, V]) -> None:
        self._data = data

    @typing.overload
    def __getitem__(self, __key: K) -> V:
        ...

    @typing.overload
    def __getitem__(self, __key: list[K]) -> Self:
        ...

    def __getitem__(self, __key: K | list[K]) -> V | Self:
        if isinstance(__key, list):
            return self.__class__({k: self._data[k] for k in __key})

        return self._data[__key]

    def __iter__(self) -> typing.Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)
