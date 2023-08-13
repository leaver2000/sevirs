from __future__ import annotations

import itertools
import json
import os
from typing import TYPE_CHECKING, Any, Final, Iterator, Literal, Mapping

import numpy as np
import polars as pl
from matplotlib.colors import Colormap, ListedColormap
from polars.type_aliases import SchemaDict

from ._typing import Array, N, Nd, cast_literal_list
from .enums import ImageEnumType

if TYPE_CHECKING:
    _mpl_colormaps: Mapping[str, Colormap]
from matplotlib import colormaps as _mpl_colormaps  # type: ignore

with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
    METADATA: Final[Mapping[str, Mapping[str, Mapping[str, Any]]]] = json.load(f)

DEFAULT_PATH_TO_SEVIR = os.getenv("PATH_TO_SEVIR", None) or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
""">>> os.environ['PATH_TO_SEVIR']"""
DEFAULT_CATALOG = "CATALOG.csv"
DEFAULT_DATA = "data"
# - Lightning Data
DEFAULT_N_FRAMES = 49  # TODO:  don't hardcode this
# Nominal Frame time offsets in minutes (used for non-raster types)
# the previous color maps and boundaries we moved to a json file
DEFAULT_FRAME_TIMES: Array[Nd[N], np.float64] = np.arange(-120.0, 125.0, 5) * 60  # in seconds
"""The lightning flashes in each from will represent the 5 minutes leading up the
the frame's time EXCEPT for the first frame, which will use the same flashes as the second frame
(This will be corrected in a future version of SEVIR so that all frames are consistent)"""
#
DEFAULT_PATCH_SIZE = int(os.getenv("PATCH_SIZE", 256))


class ColormapRegistry(Mapping[str, Colormap]):
    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Colormap]) -> None:
        self._data = data

    def __getitem__(self, item: str) -> Colormap:
        return self._data.get(item, None) or _mpl_colormaps[item]

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(self._data, _mpl_colormaps)

    def __len__(self) -> int:
        return len(self._data) + len(_mpl_colormaps)

    @classmethod
    def create(cls, *args: tuple[ImageEnumType, slice | None]) -> ColormapRegistry:
        return ColormapRegistry(
            {key.value: cls.create_cmap(key, slice_) for key, slice_ in args},
        )

    @staticmethod
    def create_cmap(img_type: ImageEnumType, slice_: slice | None = None, /, bad=0) -> Colormap:
        colors = img_type.colors
        if isinstance(colors, str):
            return _mpl_colormaps[colors]
        if slice_ is None:
            return ListedColormap(colors, name=img_type.value)
        cmap = ListedColormap(colors[slice_], name=img_type.value)
        cmap.set_extremes(bad=colors[bad], under=colors[slice_.start], over=colors[slice_.stop])
        return cmap


class ImageType(ImageEnumType):
    r"""| Sensor | Data key | Description | Spatial Resolution |  Patch Size |  Time step |
    |:--------:|:------:|:-------------:|:--------------------:|:---------:|:-----------:|
    |  GOES-16 C02 0.64 $\mu m$  |`vis`| Visible satellite imagery |  0.5 km | 768x768 | 5 minute   |
    |  GOES-16 C09 6.9 $\mu m$  |`ir069`| Infrared Satellite imagery (Water Vapor) | 2 km | 192 x 192  | 5 minutes|
    |  GOES-16 C13 10.7 $\mu m$  |`ir107`| Infrared Satellite imagery (Window) | 2 km | 192 x 192  | 5 minutes |
    |  Vertically Integrated Liquid (VIL) |`vil`|  NEXRAD radar mosaic of VIL | 1 km | 384 x 384  |5 minutes |
    |  GOES-16 GLM flashes |`lght`| Inter cloud and cloud to ground lightning events | 8 km | N/A | Continuous |
    """
    # =================================================================================================================
    VIS = "vis"
    IR_069 = "ir069"
    IR_107 = "ir107"
    VIL = "vil"
    LGHT = "lght"

    # =================================================================================================================
    # - abc interface
    def get_cmap(self) -> Colormap:
        return colormaps[self.value]

    @property
    def metadata(self) -> Mapping[str, Mapping[str, Any]]:
        return METADATA["imageType"]


IMAGE_TYPES = VIS, IR_069, IR_107, VIL, LGHT = tuple(ImageType)


colormaps: Final = ColormapRegistry.create(
    (VIL, slice(1, -1)),
    (VIS, slice(0, -1)),
    (IR_069, None),
    (IR_107, None),
    (LGHT, None),
)

# =====================================================================================================================
LightningColumn = Literal[
    0,
    1,
    2,
    3,
    4,
]
LIGHTNING_COLUMNS = FLASH_TIME, FLASH_LAT, FLASH_LON, FLASH_X, FLASH_Y = cast_literal_list(list[LightningColumn])
"""
| Column index | Meaning |
| ------------ | ------- |
| 0 | Time of flash in seconds relative to time_utc column in the catalog. |
| 1 | Reported latitude of flash in degrees |
| 2 | Reported longitude of flash in degrees |
| 3 | Flash X coordinate when converting to raster |
| 4 | Flash Y coordinate when converting to raster |

Lightning Data
The lght data is the only non-raster type in SEVIR (currently).
This data is stored in the HDF using the SEVIR id as a key.
Associated to each id is an N x 5 matrix describing each 4 hour event.
Each row of this matrix represents a single lightning flash identified by
the GOES-16 GLM sensor. The columns of this matrix are described in the following table:
"""
# - Catalog Columns
CatalogColumn = Literal[
    "id",
    "file_name",
    "file_index",
    "img_type",
    "time_utc",
    "minute_offsets",
    "episode_id",
    "event_id",
    "event_type",
    "llcrnrlat",
    "llcrnrlon",
    "urcrnrlat",
    "urcrnrlon",
    "proj",
    "size_x",
    "size_y",
    "height_m",
    "width_m",
    "data_min",
    "data_max",
    "pct_missing",
]
CATALOG_COLUMNS = (
    ID,
    FILE_NAME,
    FILE_INDEX,
    IMG_TYPE,
    TIME_UTC,
    MINUTE_OFFSETS,
    EPISODE_ID,
    EVENT_ID,
    EVENT_TYPE,
    LL_LAT,
    LL_LON,
    UR_LAT,
    UR_LON,
    PROJ,
    SIZE_X,
    SIZE_Y,
    HEIGHT_M,
    WIDTH_M,
    DATA_MIN,
    DATA_MAX,
    PCT_MISSING,
) = cast_literal_list(list[CatalogColumn])
CATALOG_BOUNDING_BOX: list[CatalogColumn] = [
    LL_LAT,
    LL_LON,
    UR_LAT,
    UR_LON,
]
CATALOG_SCHEMA: Final[SchemaDict] = {
    ID: pl.Utf8,
    FILE_NAME: pl.Utf8,
    IMG_TYPE: pl.Utf8,
    TIME_UTC: pl.Datetime,
    EPISODE_ID: pl.Float64,
    EVENT_ID: pl.Float64,
}
# one additional column for tracking file references
FILE_REF = "file_ref"
# - Event Types
EventType = Literal[
    "Hail",
    "Thunderstorm Wind",
    "Tornado",
    "Heavy Rain",
    "Flash Flood",
    "Lightning",
    "Funnel Cloud",
    "Flood",
]
EVENT_TYPES = (
    HAIL,
    THUNDERSTORM_WIND,
    TORNADO,
    HEAVY_RAIN,
    FLASH_FLOOD,
    LIGHTNING,
    FUNNEL_CLOUD,
    FLOOD,
) = cast_literal_list(list[EventType])


PROJECTION_REGEX: Final = r"""(?x)  # verbose
\+proj=(?P<projection>[a-z]+)       # projection
\s+
\+lat_0=(?P<lat>[-+]?\d{2,})        # int
\s+
\+lon_0=(?P<lon>[-+]?\d{2,})        # int
\s+
\+units=(?P<units>[a-z]+)           # m
\s+
\+a=(?P<a>[-+]?[0-9]*\.?[0-9]*)     # float
\s+
\+ellps=(?P<ellps>[a-z]+)           # WGS84
"""
"""
The projection column of the catalog looks something like.

`'+proj=laea +lat_0=38 +lon_0=-98 +units=m +a=6370997.0 +ellps=sphere '`

This is a string representation of a projection dictionary and can be used with the
>>> import polars as pl
>>> from sevir.constants import PROJECTION_REGEX
>>> df =  pl.read_csv("CATALOG.csv")
>>> df["proj"].to_pandas().str.extract(PROJECTION_REGEX)
      projection lat  lon units          a   ellps
0           laea  38  -98     m  6370997.0  sphere
1           laea  38  -98     m  6370997.0  sphere
2           laea  38  -98     m  6370997.0  sphere
3           laea  38  -98     m  6370997.0  sphere
4           laea  38  -98     m  6370997.0  sphere
...          ...  ..  ...   ...        ...     ...
75999       laea  38  -98     m  6370997.0  sphere
76000       laea  38  -98     m  6370997.0  sphere
76001       laea  38  -98     m  6370997.0  sphere
76002       laea  38  -98     m  6370997.0  sphere
76003       laea  38  -98     m  6370997.0  sphere

[76004 rows x 6 columns]
"""
