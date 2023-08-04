import argparse
from types import SimpleNamespace
from typing_extensions import Self
import typing
from typing import Any, Iterable, TypeVar, Callable

T = TypeVar("T")


@typing.overload
def argument(
    *name_or_flags: str,
    nargs: None = None,
    const: Any = None,
    default: Any = None,
    type: type[T] | None = None,
    choices: Iterable[T] | None = None,
    required: bool | None = None,
    help: str | None = None,
    metavar: str | tuple[str, ...] | None = None,
    dest: str | None = None,
    version: str | None = None,
) -> T:
    ...


@typing.overload
def argument(
    *name_or_flags: str,
    nargs: int = ...,
    const: Any = None,
    default: Any = None,
    type: type[T] | None = None,
    choices: Iterable[T] | None = None,
    required: bool | None = None,
    help: str | None = None,
    metavar: str | tuple[str, ...] | None = None,
    dest: str | None = None,
    version: str | None = None,
) -> list[T]:
    ...


def argument(
    *name_or_flags: str,
    nargs: int | None = None,
    const: Any = None,
    default: Any = None,
    type: type[T] | None = None,
    choices: Iterable[T] | None = None,
    required: bool | None = None,
    help: str | None = None,
    metavar: str | tuple[str, ...] | None = None,
    dest: str | None = None,
    version: str | None = None,
) -> T | list[T]:
    kwargs = {
        "nargs": nargs,
        "type": type,
        "const": const,
        "default": default,
        "choices": choices,
        "required": required,
        "help": help,
        "metavar": metavar,
        "dest": dest,
        "version": version,
    }

    return name_or_flags, {k: v for k, v in kwargs.items() if v is not None}  # type: ignore


class ArgumentNamespace(SimpleNamespace):
    """
    ```
    class NameSpace(ArgumentNamespace):
        foo: str
        value: list[int] = argument("value", nargs=2)

    if __name__ == "__main__":
        print(NameSpace.parse_args())
    ```
    """

    @classmethod
    def parse_args(cls) -> Self:
        parser = argparse.ArgumentParser()
        for key, value in typing.get_type_hints(cls).items():
            if key in cls.__dict__:
                names, kwargs = cls.__dict__[key]
                if "type" not in kwargs:
                    if iterable := typing.get_args(value):
                        kwargs["type"] = iterable[0]
                    else:
                        kwargs["type"] = value
                parser.add_argument(*names, **kwargs)
            else:
                parser.add_argument(f"--{key}", type=value)

        return cls(**vars(parser.parse_args()))
