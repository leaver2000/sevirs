# flake8: noqa
"""
Type hints for the sevir package.

Note:
-----
To avoid circular imports if this module needs to import anything from sevir,
the import should be accomplished conditionally under `TYPE_CHECKING` ie:

```
if TYPE_CHECKING:
    from .core.catalog import Catalog
```
"""
from __future__ import annotations

import enum
import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    Sized,
    TypeAlias,
    TypedDict,
    TypeVar,
    get_args,
)

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import Colormap, Normalize
from pandas._typing import Scalar

if sys.version_info < (3, 11):
    from typing_extensions import Self, TypeVarTuple, Unpack
else:
    from typing import Self, TypeVarTuple, Unpack
if TYPE_CHECKING:  # avoid circular imports
    from .constants import ImageType as _ImageType
    from .core.catalog import Catalog as _Catalog
else:
    _Catalog = Any
    _ImageType = Any


# =====================================================================================================================
_T1_co = TypeVar("_T1_co", covariant=True)
_T2_co = TypeVar("_T2_co", covariant=True)


Ts = TypeVarTuple("Ts")
AnyT = TypeVar("AnyT", bound=Any)
KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
ScalarT = TypeVar("ScalarT", bound=Scalar)

DictStr: TypeAlias = dict[str, AnyT]
DictStrAny: TypeAlias = DictStr[Any]
StrPath: TypeAlias = str | os.PathLike[str]
CatalogData: TypeAlias = "_Catalog | pl.DataFrame | pd.DataFrame | StrPath"
PatchSize: TypeAlias = int | Literal["upscale", "downscale"]

ImageName: TypeAlias = "Literal['vis', 'vil', 'ir069', 'ir107', 'lght'] | _ImageType"
ImageLike: TypeAlias = "_ImageType | ImageName"
ImageTypes: TypeAlias = "tuple[_ImageType, ...]"
ImageSequence: TypeAlias = "Sequence[ImageName]"


def cast_literal_list(cls: type[AnyT]) -> AnyT:
    """
    >>> Numbers = typing.Literal[1, 2, 3]
    >>> NUM_LIST = ONE, TWO, THREE = cast_literal_list(list[Numbers])
    >>> NUM_LIST
    [1, 2, 3]
    """
    (literal,) = get_args(cls)
    values = get_args(literal)
    return list(values)  # type: ignore[return-value]


# =====================================================================================================================
class Nd(Generic[Unpack[Ts]]):
    """A declarative class for annotating the Number of dimensions in a
    `numpy` array.

    Example:
    --------
    >>> import numpy as np
    >>> from sevir._typing import Nd
    >>> a: np.ndarray[Nd[2, 2], np.int64] = np.array([[1, 2], [3, 4]])
    """


N = enum.Enum(":", {"_": slice(None)})
_NdT = TypeVar("_NdT", bound=Nd, contravariant=True)
Array: TypeAlias = np.ndarray[_NdT, np.dtype[AnyT]]
"""
>>> from typing_extensions import reveal_type
>>> import numpy as np
>>> from sevir._typing import Array, Nd, N
>>> a: Array[Nd[N, N], np.int64] = np.array([[1, 2], [3, 4]])
>>> reveal_type(a)
Runtime type is 'ndarray'
"""


# =====================================================================================================================


class ImageKwargs(TypedDict, total=False):
    """partial config passed to `matplotlib.pyplot.imshow`"""

    aspect: float | Literal["equal", "auto"]
    interpolation: str
    alpha: float | np.ndarray
    origin: Literal["upper", "lower"]
    extent: Sequence[float]
    interpolation_stage: Literal["data", "rgba"]


class ImageConfig(ImageKwargs, total=False):
    """partial config passed to `matplotlib.pyplot.imshow`"""

    cmap: Colormap
    norm: Normalize


# =====================================================================================================================
# - Protocols
# =====================================================================================================================
class Shaped(Sized, Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...


class Closeable(Protocol):
    def close(self) -> None:
        ...


class FrameProtocol(Shaped, Generic[_T1_co, _T2_co], Protocol):
    @property
    def columns(self) -> _T1_co:
        ...

    @property
    def dtypes(self) -> _T2_co:
        ...


class EnumProtocol(Protocol[AnyT]):
    value: AnyT
    __iter__: Callable[..., Iterable[Self]]

    @classmethod
    def __len__(cls) -> int:
        ...

    @classmethod
    def __next__(cls) -> Self:
        ...

    @classmethod
    def __getitem__(cls, name: str) -> Self:
        ...

    @classmethod
    def __call__(cls, value: Any) -> Self:
        ...
