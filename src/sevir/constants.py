import os
from typing import Literal

import numpy as np

from ._typing import cast_literal_list

DATA_INDEX = "data_index"
DEFAULT_PATH_TO_SEVIR = os.getenv("PATH_TO_SEVIR", None) or os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOG = "CATALOG.csv"
DEFAULT_DATA = "data"

# - Lightning Data
DEFAULT_N_FRAMES = 49  # TODO:  don't hardcode this
# Nominal Frame time offsets in minutes (used for non-raster types)

DEFAULT_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
"""The lightning flashes in each from will represent the 5 minutes leading up the
the frame's time EXCEPT for the first frame, which will use the same flashes as the second frame
(This will be corrected in a future version of SEVIR so that all frames are consistent)"""
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


# - Image Types
ImageType = Literal[
    "vis",
    "ir069",
    "ir107",
    "vil",
    "lght",
]
IMAGE_TYPES = VIS, IR_069, IR_107, VIL, LGHT = cast_literal_list(list[ImageType])

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
