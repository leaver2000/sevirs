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
    Generic,
    Hashable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
    get_args,
)

import numpy as np
import pandas as pd
import polars as pl
from pandas._typing import Scalar

if sys.version_info < (3, 11):
    from typing_extensions import TypeVarTuple, Unpack
else:
    from typing import TypeVarTuple, Unpack
if TYPE_CHECKING:  # avoid circular imports
    from .constants import ImageType as _ImageType
    from .core.catalog import Catalog as _Catalog
else:
    _Catalog = Any
    _ImageType = Any


# =====================================================================================================================
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
from typing import TypedDict

from matplotlib.colors import Colormap, Normalize


class ImageShowConfig(TypedDict, total=False):
    # X: Unknown,
    cmap: str | Colormap | None
    norm: Normalize | None
    aspect: float | Literal["equal", "auto"] | None
    interpolation: str | None
    alpha: float | np.ndarray | None
    vmin: float | None
    vmax: float | None
    origin: Literal["upper", "lower"] | None
    extent: Sequence[float] | None

    interpolation_stage: Literal["data", "rgba"] | None
    filternorm: bool
    filterrad: float
    resample: bool | None
    url: str | None
    # **kwargs: Unknown
