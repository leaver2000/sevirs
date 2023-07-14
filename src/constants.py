import enum
import os

os.environ["PATH_TO_SEVIR"] = "/mnt/nuc/c/sevir"

import typing

import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import Self

_ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_SEVIR = os.getenv("PATH_TO_SEVIR", _ROOT_DIR)
DEFAULT_CATALOG = "CATALOG.csv"
DEFAULT_DATA = "data"
DEFAULT_N_FRAMES = 49  # TODO:  don't hardcode this
# Nominal Frame time offsets in minutes (used for non-raster types)

DEFAULT_FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
"""The lightning flashes in each from will represent the 5 minutes leading up the
the frame's time EXCEPT for the first frame, which will use the same flashes as the second frame
(This will be corrected in a future version of SEVIR so that all frames are consistent)"""


class Enum(enum.Enum):
    @classmethod
    def _missing_(cls, value: object) -> typing.Any:
        return cls.__members__[str(value).upper()]

    @classmethod
    def map(cls, __values: typing.Iterable[typing.Any], /) -> list[Self]:
        """class method to map values to enum members"""
        return [cls(value) for value in ([__values] if isinstance(__values, (str, Enum)) else __values)]


# =====================================================================================================================
class ImageType(str, Enum):
    VISIBLE = "vis"
    IR_069 = "ir069"
    IR_107 = "ir107"
    VERTICALLY_INTEGRATED_LIQUID = "vil"
    LIGHTNING = "lght"

    def get_dtype(self) -> DTypeLike:
        return SEVIR_DTYPES[self]

    def get_cmap(self) -> typing.Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"


ImageTypeValue = typing.Literal["vis", "ir069", "ir107", "vil", "lght"]
ImageTypeLike: typing.TypeAlias = "ImageType | ImageTypeValue"
ImageTypeSet: typing.TypeAlias = "set[ImageType] | set[ImageTypeValue] | set[ImageTypeLike]"
DEFAULT_IMAGE_TYPES = set(ImageType)

VISIBLE, IR_069, IR_107, VERTICALLY_INTEGRATED_LIQUID, LIGHTNING = (
    ImageType.VISIBLE,
    ImageType.IR_069,
    ImageType.IR_107,
    ImageType.VERTICALLY_INTEGRATED_LIQUID,
    ImageType.LIGHTNING,
)

# =====================================================================================================================
SEVIR_DTYPES: dict[ImageType, DTypeLike] = {
    VERTICALLY_INTEGRATED_LIQUID: np.uint8,
    VISIBLE: np.int16,
    IR_069: np.int16,
    IR_107: np.int16,
    LIGHTNING: np.int16,
}


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
) = (
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
)
CATALOG_BOUNDING_BOX = [
    LL_LAT,
    LL_LON,
    UR_LAT,
    UR_LON,
]
CATALOG_DTYPES = {
    ID: "string",
    FILE_NAME: "string",
    IMG_TYPE: "string",
    PROJ: "string",
    MINUTE_OFFSETS: "string",
    EVENT_TYPE: "string",
    EVENT_ID: "Int64",
    EPISODE_ID: "Int64",
    LL_LAT: "float",
    LL_LON: "float",
    UR_LAT: "float",
    UR_LON: "float",
}
# =====================================================================================================================
EventType: typing.TypeAlias = typing.Literal[
    "Hail",
    "Thunderstorm Wind",
    "Tornado",
    "Heavy Rain",
    "Flash Flood",
    "Lightning",
    "Funnel Cloud",
    "Flood",
]

EVENTS: list[EventType] = list(typing.get_args(EventType))
Hail, ThunderstormWind, Tornado, HeavyRain, FlashFlood, Lightning, FunnelCloud, Flood = EVENTS
