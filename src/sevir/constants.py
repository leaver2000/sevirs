from __future__ import annotations

import enum
import json
import logging
import os
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import polars as pl

from ._typing import cast_literal_list

logging.getLogger().setLevel(logging.INFO)
DEFAULT_PATH_TO_SEVIR = os.getenv("PATH_TO_SEVIR", None) or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOG = "CATALOG.csv"
DEFAULT_DATA = "data"
# - Lightning Data
DEFAULT_N_FRAMES = 49  # TODO:  don't hardcode this
# Nominal Frame time offsets in minutes (used for non-raster types)
# the previous color maps and boundaries we moved to a json file
DEFAULT_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
"""The lightning flashes in each from will represent the 5 minutes leading up the
the frame's time EXCEPT for the first frame, which will use the same flashes as the second frame
(This will be corrected in a future version of SEVIR so that all frames are consistent)"""
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
    def __new__(cls, value: str, sensor: str, description: str, patch_size: int, time_steps: int) -> ImageInfo:
        ...

    @overload
    def __new__(cls, value: ImageInfo) -> ImageInfo:
        ...

    def __new__(
        cls,
        value: str | ImageInfo,
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
        return pl.col(self)


class ImageType(ImageInfo, enum.Enum):
    r"""| Sensor | Data key | Description | Spatial Resolution |  Patch Size |  Time step |
    |:--------:|:------:|:-------------:|:--------------------:|:---------:|:-----------:|
    |  GOES-16 C02 0.64 $\mu m$  |`vis`| Visible satellite imagery |  0.5 km | 768x768 | 5 minute   |
    |  GOES-16 C09 6.9 $\mu m$  |`ir069`| Infrared Satellite imagery (Water Vapor) | 2 km | 192 x 192  | 5 minutes|
    |  GOES-16 C13 10.7 $\mu m$  |`ir107`| Infrared Satellite imagery (Window) | 2 km | 192 x 192  | 5 minutes |
    |  Vertically Integrated Liquid (VIL) |`vil`|  NEXRAD radar mosaic of VIL | 1 km | 384 x 384  |5 minutes |
    |  GOES-16 GLM flashes |`lght`| Inter cloud and cloud to ground lightning events | 8 km | N/A | Continuous |
    """
    description: str
    sensor: str
    patch_size: int
    time_steps: int
    VIS = ("vis", "GOES-16 C02 0.64", "Visible satellite imagery", 768, 5)
    IR_069 = ("ir069", "GOES-16 C13 6.9", "Infrared Satellite imagery (Water Vapor)", 192, 5)
    IR_107 = ("ir107", "GOES-16 C13 10.7", "Infrared Satellite imagery (Window)", 192, 5)
    VIL = ("vil", "Vertically Integrated Liquid (VIL)", "NEXRAD radar mosaic of VIL", 384, 5)
    LGHT = ("lght", "GOES-16 GLM flashes", "Detections of inter cloud and cloud to ground lightning events", 0, 0)

    def __str__(self) -> str:
        return str(self)


IMAGE_TYPES = VIS, IR_069, IR_107, VIL, LGHT = list(ImageType)

# =====================================================================================================================
LightningColumns = Literal[
    0,
    1,
    2,
    3,
    4,
]

LIGHTNING_COLUMNS = FLASH_TIME, FLASH_LAT, FLASH_LON, FLASH_X, FLASH_Y = cast_literal_list(list[LightningColumns])
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
