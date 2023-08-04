from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING, Any, Collection, Final

import pandas as pd
import polars as pl
from polars.dataframe.frame import DataFrame
from polars.type_aliases import IntoExpr
from typing_extensions import Self

from .constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_PATH_TO_SEVIR,
    EPISODE_ID,
    EVENT_ID,
    EVENT_TYPE,
    FILE_INDEX,
    FILE_NAME,
    FILE_REF,
    ID,
    IMAGE_TYPES,
    IMG_TYPE,
    TIME_UTC,
    EventType,
    ImageType,
)
from .generic import PolarsAdapter


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
class AbstractCatalog(abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> pl.DataFrame:
        ...

    @property
    def columns(self) -> list[str]:
        return self.data.columns

    @property
    def id(self) -> pl.Series:  # noqa: A003
        return self.data[ID]

    @property
    def file_name(self) -> pl.Series:
        return self.data[FILE_NAME]

    @property
    def file_index(self) -> pl.Series:
        return self.data[FILE_INDEX]

    @property
    def file_ref(self) -> pl.Series:
        return self.data[FILE_REF]

    @property
    def image_types(self) -> pl.Series:
        return self.data[IMG_TYPE]

    @property
    def event_type(self) -> pl.Series:
        return self.data[EVENT_TYPE]

    @property
    def time_utc(self) -> pl.Series:
        return self.data[TIME_UTC]


class Catalog(PolarsAdapter, AbstractCatalog):
    __slots__ = ("types",)
    if TYPE_CHECKING:
        types: Final[tuple[ImageType, ...]]

    # - Initialization
    def _manager(
        self, data: DataFrame, *, inplace: bool, img_types: tuple[ImageType, ...] | None = None, **kwargs: Any
    ) -> Self:
        return super()._manager(data, inplace=inplace, img_types=img_types or self.types, **kwargs)

    @staticmethod
    def _constructor(
        data: pl.DataFrame | pd.DataFrame | Catalog | str = DEFAULT_PATH_TO_SEVIR,
        *,
        img_types: tuple[ImageType, ...] | None = None,
        catalog: str | None = DEFAULT_CATALOG,
        data_dir: str | None = DEFAULT_DATA,
    ) -> tuple[pl.DataFrame, tuple[ImageType, ...]]:
        if isinstance(data, Catalog):
            return data.data, data.types
        # validation
        if img_types is not None and len(img_types) != len(set(img_types)):
            raise ValueError("Duplicate image types in img_types")

        # construction
        if isinstance(data, str) and catalog is not None and data_dir is not None:
            data = read(
                os.path.join(data, catalog),
                os.path.join(data, data_dir),
                img_types=list(set(img_types or IMAGE_TYPES)),
            )
        elif isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
        elif isinstance(data, pl.DataFrame):
            data = data
        else:
            if isinstance(data, str) and (catalog is None or data_dir is None):
                raise ValueError("catalog and data_dir cannot be None when data is a path")
            raise ValueError(f"Invalid type for data: {type(data)}")

        return data.with_columns(**{FILE_REF: None}), img_types or tuple(IMAGE_TYPES)

    def __init__(
        self,
        data: pl.DataFrame | pd.DataFrame | Catalog | str = DEFAULT_PATH_TO_SEVIR,
        *,
        img_types: tuple[ImageType, ...] | None = None,
        catalog: str | None = DEFAULT_CATALOG,
        data_dir: str | None = DEFAULT_DATA,
    ) -> None:
        data, types = self._constructor(data, img_types=img_types, catalog=catalog, data_dir=data_dir)
        super().__init__(data)
        self.types = types

    # - Methods
    def _where(
        self,
        col: str,
        value: EventType | ImageType | Collection[EventType | ImageType],
    ) -> pl.DataFrame:
        s = self.data[col]
        return self.data.filter(s.is_in(value)) if not isinstance(value, str) else self.data.filter(s == value)

    def where(
        self,
        col: str,
        value: EventType | ImageType | Collection[EventType | ImageType],
        inplace: bool = False,
    ) -> Catalog:
        return self._manager(self._where(col, value), inplace=inplace)

    def get_by_img_type(self, img_types: ImageType | Collection[ImageType], inplace=False) -> Catalog:
        return self._manager(
            self._where(IMG_TYPE, img_types),
            img_types=tuple((img_types,) if isinstance(img_types, str) else img_types),
            inplace=inplace,
        )

    def get_by_event(self, event: EventType | Collection[EventType], inplace: bool = False) -> Catalog:
        return self.where(EVENT_TYPE, event, inplace=inplace)

    def split_by_types(self, x: list[ImageType], y: list[ImageType]) -> tuple[Catalog, Catalog]:
        if not set(x + y) <= set(self.image_types):
            raise ValueError(
                f"Catalog does not contain all of the requested image types: {set(x + y) - set(self.image_types)}."
                " Use `Catalog.get_by_img_type` to get a subset of the catalog."
            )

        return self.get_by_img_type(x), self.get_by_img_type(y)

    # =================================================================================================================
    # - File Reference

    def with_reference(self, ref: IntoExpr, inplace=False) -> Catalog:
        return self._manager(
            self.data.with_columns(**{FILE_REF: ref}),
            inplace=inplace,
        )

    def close(self) -> None:
        self._manager(
            self.data.with_columns(**{FILE_REF: None}),
            inplace=True,
        )
