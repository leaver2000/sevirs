from __future__ import annotations

import os
from typing import Any, Collection, Final

import numpy as np
import pandas as pd
import polars as pl
from polars.dataframe.frame import DataFrame
from polars.type_aliases import IntoExpr
from typing_extensions import Self

from .._typing import CatalogData, StrPath
from ..constants import (
    CATALOG_SCHEMA,
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_PATH_TO_SEVIR,
    EVENT_TYPE,
    FILE_NAME,
    FILE_REF,
    ID,
    IMAGE_TYPES,
    IMG_TYPE,
    PROJ,
    PROJECTION_REGEX,
    CatalogColumn,
    EventType,
    ImageType,
)
from ..generic import AbstractCatalog, PolarsAdapter


# =====================================================================================================================
def read(
    __src: StrPath = DEFAULT_PATH_TO_SEVIR,
    /,
    *,
    catalog: StrPath | None = None,
    data: StrPath | None = None,
    img_types: Collection[ImageType] | None = None,
    extract_projection: bool = False,
) -> pl.DataFrame:
    """
    ```
    import sevir.core.catalog
    df = sevir.core.catalog.read("/path/to/sevir")
    df = sevir.core.catalog.read("/path/to/sevir/CATALOG.csv")
    ```

    just set the environment variable `PATH_TO_SEVIR` to the path to the SEVIR data directory.
    ```
    import os
    os.environ["PATH_TO_SEVIR"] = "/path/to/sevir"
    import sevir.core.catalog
    df = sevir.core.catalog.read()
    ```
    """
    # - resolve paths
    if os.path.isdir(__src):
        src_catalog = os.path.join(__src, catalog or DEFAULT_CATALOG)
        src_data = os.path.join(__src, data or DEFAULT_DATA)
    elif os.path.isfile(__src):
        src_catalog = os.path.abspath(__src)
        src_data = os.path.join(os.path.dirname(__src), data or DEFAULT_DATA)
    else:
        raise FileNotFoundError(f"Invalid path to SEVIR data: {__src}")

    if not os.path.isfile(src_catalog):
        raise FileNotFoundError(f"Catalog file not found: {catalog}")

    elif not os.path.isdir(src_data):
        raise FileNotFoundError(f"SEVIR data directory not found: {src_data}")

    # - read catalog
    df = pl.read_csv(src_catalog, dtypes=CATALOG_SCHEMA, null_values=[""], use_pyarrow=True)
    df = df if img_types is None else subset(df, img_types)
    df = df.with_columns(df[FILE_NAME].apply(lambda s: os.path.join(src_data, s)))

    if extract_projection:
        proj = pl.from_pandas(
            df[PROJ].to_pandas().str.extract(PROJECTION_REGEX).astype({"lat": float, "lon": float, "a": float})
        )
        df = pl.concat([df, proj], how="horizontal")

    return df


def subset(df: pl.DataFrame, img_types: Collection[ImageType]) -> pl.DataFrame:
    df = df.filter(df[IMG_TYPE].is_in(img_types))
    count = df.groupby(ID).count()
    count = count.filter(count["count"] >= len(img_types))
    return df.filter(df[ID].is_in(count[ID]))


# =====================================================================================================================
class Catalog(PolarsAdapter, AbstractCatalog):
    __slots__ = ("types",)

    # - Initialization
    def _manager(  # type: ignore[override]
        self, data: DataFrame, *, inplace: bool, img_types: tuple[ImageType, ...] | None = None, **kwargs: Any
    ) -> Self:
        return super()._manager(data, inplace=inplace, img_types=img_types or self.types, **kwargs)

    @staticmethod
    def _constructor(
        src: CatalogData = DEFAULT_PATH_TO_SEVIR,
        img_types: tuple[ImageType, ...] | None = None,
        catalog: str | None = None,
        data_dir: str | None = None,
    ) -> tuple[pl.DataFrame, tuple[ImageType, ...]]:
        # - fast path
        if isinstance(src, Catalog):
            return src.data, src.types

        # - validation
        if img_types is not None and len(img_types) != len(set(img_types)):
            raise ValueError("Duplicate image types in img_types")

        img_types = img_types or IMAGE_TYPES

        # - construction
        if isinstance(src, str):
            data = read(src, catalog=catalog, data=data_dir, img_types=img_types)
        elif isinstance(src, pd.DataFrame):
            data = pl.from_pandas(src)
        elif isinstance(src, pl.DataFrame):
            data = src
        else:
            raise ValueError(f"Invalid type for data: {type(src)}")

        # TODO: add a check for columns
        return data, img_types

    def __init__(
        self,
        data: CatalogData = DEFAULT_PATH_TO_SEVIR,
        /,
        *,
        img_types: tuple[ImageType, ...] | None = None,
        catalog: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        # - unpack
        data, types = self._constructor(data, img_types, catalog, data_dir)

        # - copy the data and initialize the catalog
        super().__init__(data.with_columns(**{FILE_REF: None}))
        self.types: Final[tuple[ImageType, ...]] = types

    # - Methods
    def _where(
        self, col: CatalogColumn, value: EventType | ImageType | Collection[EventType | ImageType]
    ) -> pl.DataFrame:
        series = self.data[col]
        predicate = series.is_in(value) if not isinstance(value, str) else series == value
        return self.data.filter(predicate)

    def where(
        self,
        col: CatalogColumn,
        value: EventType | ImageType | Collection[EventType | ImageType],
        inplace: bool = False,
    ) -> Catalog:
        df = self._where(col, value)
        return self._manager(df, inplace=inplace)

    def get_by_img_type(self, img_types: ImageType | Collection[ImageType], inplace=False) -> Catalog:
        return self._manager(
            self._where(IMG_TYPE, img_types),
            img_types=tuple((img_types,) if isinstance(img_types, str) else img_types),
            inplace=inplace,
        )

    def get_by_event(self, event: EventType | Collection[EventType], inplace: bool = False) -> Catalog:
        return self.where(EVENT_TYPE, event, inplace=inplace)

    def intersect(self, x: tuple[ImageType, ...], y: tuple[ImageType, ...]) -> tuple[Catalog, Catalog]:
        if not set(x + y) <= set(self.img_type):
            raise ValueError(
                f"Catalog does not contain all of the requested image types: {set(x + y) - set(self.img_type)}."
                " Use `Catalog.get_by_img_type` to get a subset of the catalog."
            )

        df_x, df_y = (self._where(IMG_TYPE, img_types) for img_types in (x, y))
        ids = set(np.intersect1d(df_x[ID].to_numpy(), df_y[ID].to_numpy()))  # type: set[str]
        df_x, df_y = (df.filter(df[ID].is_in(ids)) for df in (df_x, df_y))

        return (
            self._manager(df_x, inplace=False, img_types=x),
            self._manager(df_y, inplace=False, img_types=y),
        )

    def with_projection(self) -> pl.DataFrame:
        from .plot import parse_projection

        return pl.concat(
            [self.data.drop([PROJ]), pl.DataFrame(self.data[PROJ].apply(parse_projection).to_list())],
            how="horizontal",
        )

    # =================================================================================================================
    # - File Reference
    def with_reference(self, ref: IntoExpr, *, inplace=False) -> Catalog:
        return self._manager(
            self.data.with_columns(**{FILE_REF: ref}),
            inplace=inplace,
        )

    def close(self) -> None:
        self.with_reference(None, inplace=True)
