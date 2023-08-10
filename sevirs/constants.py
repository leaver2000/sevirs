from __future__ import annotations

import enum
import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, overload

import numpy as np
import polars as pl
from polars.type_aliases import SchemaDict

from ._typing import Array, N, Nd, cast_literal_list

logging.getLogger().setLevel(logging.INFO)
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


with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
    CONFIG = json.load(f)


# - Image Types
class ImageInfo(str):
    __slots__ = ("_value_", "description", "sensor", "patch_size", "time_steps")
    if TYPE_CHECKING:
        _value_: str
        description: str
        sensor: str
        patch_size: int
        time_steps: int

    @overload
    def __new__(
        cls,
        value: str,
        sensor: str,
        description: str,
        patch_size: int,
        time_steps: int,
    ) -> ImageInfo:
        ...

    @overload
    def __new__(
        cls,
        value: ImageInfo | Literal["vis", "vil", "ir069", "ir107", "lght"],
    ) -> ImageInfo:
        ...

    def __new__(
        cls,
        value: ImageInfo | Literal["vis", "vil", "ir069", "ir107", "lght"] | str,
        sensor: str | None = None,
        description: str | None = None,
        patch_size: int | None = None,
        time_steps: int | None = None,
    ) -> ImageInfo:
        if isinstance(value, ImageInfo):
            sensor = value.sensor
            description = value.description
            patch_size = value.patch_size
            time_steps = value.time_steps
            value = value._value_
        else:
            assert isinstance(description, str)
            assert isinstance(sensor, str)
            assert isinstance(patch_size, int)
            assert isinstance(time_steps, int)

        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        obj.sensor = sensor
        obj.patch_size = patch_size
        obj.time_steps = time_steps
        return obj

    def col(self):
        return pl.col(self._value_)


class ImageType(ImageInfo, enum.Enum):
    r"""| Sensor | Data key | Description | Spatial Resolution |  Patch Size |  Time step |
    |:--------:|:------:|:-------------:|:--------------------:|:---------:|:-----------:|
    |  GOES-16 C02 0.64 $\mu m$  |`vis`| Visible satellite imagery |  0.5 km | 768x768 | 5 minute   |
    |  GOES-16 C09 6.9 $\mu m$  |`ir069`| Infrared Satellite imagery (Water Vapor) | 2 km | 192 x 192  | 5 minutes|
    |  GOES-16 C13 10.7 $\mu m$  |`ir107`| Infrared Satellite imagery (Window) | 2 km | 192 x 192  | 5 minutes |
    |  Vertically Integrated Liquid (VIL) |`vil`|  NEXRAD radar mosaic of VIL | 1 km | 384 x 384  |5 minutes |
    |  GOES-16 GLM flashes |`lght`| Inter cloud and cloud to ground lightning events | 8 km | N/A | Continuous |
    """
    if TYPE_CHECKING:
        _member_map_: ClassVar[dict[str, ImageType]]
        value: str
        description: str
        sensor: str
        patch_size: int
        time_steps: int

    VIS = ImageInfo("vis", "GOES-16 C02 0.64", "Visible satellite imagery", 768, 5)
    IR_069 = ImageInfo("ir069", "GOES-16 C13 6.9", "Infrared Satellite imagery (Water Vapor)", 192, 5)
    IR_107 = ImageInfo("ir107", "GOES-16 C13 10.7", "Infrared Satellite imagery (Window)", 192, 5)
    VIL = ImageInfo("vil", "Vertically Integrated Liquid (VIL)", "NEXRAD radar mosaic of VIL", 384, 5)
    LGHT = ImageInfo(
        "lght", "GOES-16 GLM flashes", "Detections of inter cloud and cloud to ground lightning events", 0, 0
    )

    def __str__(self) -> str:
        return str(self._value_)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def _missing_(cls, value: Any) -> ImageType:
        if result := cls._member_map_.get(value, None):  # type: ignore[attr-defined]
            return result
        raise ValueError(f"{value} is not a valid {cls.__name__}")


IMAGE_TYPES = VIS, IR_069, IR_107, VIL, LGHT = tuple(ImageType)
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
