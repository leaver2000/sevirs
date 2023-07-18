from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import polars as pl

from ._adapters import PolarsAdapter
from .constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_PATH_TO_SEVIR,
    EPISODE_ID,
    EVENT_ID,
    EVENT_TYPE,
    FILE_INDEX,
    FILE_NAME,
    ID,
    IMAGE_TYPES,
    IMG_TYPE,
    IR_069,
    IR_107,
    TIME_UTC,
    VIS,
    EventType,
    ImageType,
)

# =====================================================================================================================


def read(
    catalog: str = os.path.join(DEFAULT_PATH_TO_SEVIR, DEFAULT_CATALOG),
    sevir_data: str = os.path.join(DEFAULT_PATH_TO_SEVIR, DEFAULT_DATA),
    /,
    *,
    img_types: list[ImageType] | None = None,
) -> pl.DataFrame:
    df = pl.read_csv(
        catalog,
        dtypes={
            ID: pl.Utf8,
            FILE_NAME: pl.Utf8,
            IMG_TYPE: pl.Utf8,
            TIME_UTC: pl.Datetime,
            EPISODE_ID: pl.Float64,
            EVENT_ID: pl.Float64,
        },
        null_values=[""],
        use_pyarrow=True,
    )
    df = df if img_types is None else subset_by_image_types(df, img_types)

    return df.with_columns(
        df[FILE_NAME].apply(lambda s: os.path.join(sevir_data, s)),
    )


def subset_by_image_types(df: pl.DataFrame, img_types: list[ImageType]) -> pl.DataFrame:
    df = df.filter(df[IMG_TYPE].is_in(img_types))
    count = df.groupby(ID).count()
    count = count.filter(count["count"] >= len(img_types))
    return df.filter(df[ID].is_in(count[ID]))


# =====================================================================================================================


class Catalog(PolarsAdapter):
    def __init__(
        self,
        data: pl.DataFrame | pd.DataFrame | Catalog | str = DEFAULT_PATH_TO_SEVIR,
        *,
        img_types: Iterable[ImageType] | None = None,
        catalog: str | None = DEFAULT_CATALOG,
        data_dir: str | None = DEFAULT_DATA,
    ) -> None:
        if isinstance(data, str) and catalog is not None and data_dir is not None:
            df = read(
                os.path.join(data, catalog),
                os.path.join(data, data_dir),
                img_types=list(set(img_types or IMAGE_TYPES)),
            )

        elif isinstance(data, pd.DataFrame):
            df = pl.from_pandas(data)
        elif isinstance(data, Catalog):
            df = data.data
        elif isinstance(data, pl.DataFrame):
            df = data
        else:
            if isinstance(data, str) and (catalog is None or data_dir is None):
                raise ValueError("catalog and data_dir cannot be None when data is a path")
            raise ValueError(f"Invalid type for data: {type(data)}")

        super().__init__(df)

        self._repr_meta_ = ", ".join(self.image_set)

    # - Properties

    @property
    def id(self) -> pl.Series:  # noqa
        return self.data[ID]

    @property
    def file_name(self) -> pl.Series:
        return self.data[FILE_NAME]

    @property
    def file_index(self) -> pl.Series:
        return self.data[FILE_INDEX]

    @property
    def image_type(self) -> pl.Series:
        return self.data[IMG_TYPE]

    @property
    def image_set(self) -> set[ImageType]:
        return set(self.image_type)

    @property
    def event_type(self) -> pl.Series:
        return self.data[EVENT_TYPE]

    @property
    def time_utc(self) -> pl.Series:
        return self.data[TIME_UTC]

    # =================================================================================================================
    # - Methods

    def where(
        self,
        col: str,
        value: EventType | ImageType | Iterable[EventType | ImageType],
        inplace: bool = False,
    ) -> Catalog:
        s = self.data[col]
        df = self.data.filter(s.is_in(value)) if isinstance(value, list) else self.data.filter(s == value)
        return self._manager(df, inplace=inplace)

    def get_by_img_type(self, img_types: ImageType | Iterable[ImageType], inplace=False) -> Catalog:
        return self.where(IMG_TYPE, img_types, inplace=inplace)

    def get_by_event(self, event: EventType | Iterable[EventType], inplace: bool = False) -> Catalog:
        return self.where(EVENT_TYPE, event, inplace=inplace)

    def split_by_types(self, x: list[ImageType], y: list[ImageType]) -> tuple[Catalog, Catalog]:
        img_types = self.image_set

        if not set(x + y).issubset(img_types):
            raise ValueError(
                f"Catalog does not contain all of the requested image types: {set(x + y) - img_types}."
                " Use `Catalog.get_by_img_type` to get a subset of the catalog."
            )

        return self.get_by_img_type(x), self.get_by_img_type(y)


def main() -> None:
    cat = Catalog(img_types=[VIS, IR_069, IR_107])
    x, y = cat.split_by_types([VIS, IR_069], [IR_107])
    print(x, y)


if __name__ == "__main__":
    main()
