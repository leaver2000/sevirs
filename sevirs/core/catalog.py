from __future__ import annotations

import os
from typing import Any, Collection, Final

import numpy as np
import pandas as pd
import polars as pl
from polars.type_aliases import IntoExpr
from typing_extensions import Self

from .._typing import CatalogData, ImageLike, ImageSequence, ImageTypes, StrPath
from ..constants import (
    CATALOG_SCHEMA,
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_PATH_TO_SEVIR,
    EVENT_TYPE,
    FILE_NAME,
    FILE_REF,
    ID,
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
    img_types: ImageSequence | None = None,
    extract_projection: bool = False,
    drop_duplicates: bool = True,
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

    # - validate paths
    if not os.path.isfile(src_catalog):
        raise FileNotFoundError(f"Catalog file not found: {catalog}")
    elif not os.path.isdir(src_data):
        raise FileNotFoundError(f"SEVIR data directory not found: {src_data}")

    # - read catalog
    df = pl.read_csv(src_catalog, dtypes=CATALOG_SCHEMA, null_values=[""], use_pyarrow=True)
    df = df if img_types is None else subset(df, img_types)
    df = df.with_columns(df[FILE_NAME].apply(lambda s: os.path.join(src_data, s)))

    if drop_duplicates:
        # TODO: rather than dropping the ID's the images should be joined together in the x/y dimension
        # or a new id should be created
        df = df.groupby([ID, IMG_TYPE]).first()

    if extract_projection:
        proj = pl.from_pandas(
            df[PROJ].to_pandas().str.extract(PROJECTION_REGEX).astype({"lat": float, "lon": float, "a": float})
        )
        df = pl.concat([df, proj], how="horizontal")

    return df


def subset(df: pl.DataFrame, img_types: ImageSequence) -> pl.DataFrame:
    df = df.filter(df[IMG_TYPE].is_in(img_types))
    count = df.groupby(ID).count()
    count = count.filter(count["count"] >= len(img_types))
    return df.filter(df[ID].is_in(count[ID]))


# =====================================================================================================================
class Catalog(PolarsAdapter, AbstractCatalog):
    __slots__ = ("types",)

    # - Initialization
    def _manager(  # type: ignore[override]
        self, data: pl.DataFrame, *, inplace: bool, img_types: ImageSequence | None = None, **kwargs: Any
    ) -> Self:
        return super()._manager(data, inplace=inplace, img_types=img_types or self.types, **kwargs)

    @staticmethod
    def _constructor(
        src: CatalogData,
        img_types: ImageSequence | None,
        catalog: str | None,
        data_dir: str | None,
        drop_duplicates: bool,
    ) -> tuple[pl.DataFrame, ImageTypes]:
        # - fast path
        if isinstance(src, Catalog):
            return src.data, src.types
        if img_types is not None and len(img_types) != len(set(img_types)):
            raise ValueError("Duplicate image types are not allowed.")
        else:
            img_types = ImageType.sequential(img_types)  # type: ignore[misc]

        # - construction
        if isinstance(src, str):
            data = read(src, catalog=catalog, data=data_dir, img_types=img_types, drop_duplicates=drop_duplicates)
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
        img_types: ImageSequence | None = None,
        catalog: str | None = None,
        data_dir: str | None = None,
        drop_duplicates: bool = True,
    ) -> None:
        data, types = self._constructor(data, img_types, catalog, data_dir, drop_duplicates)

        super().__init__(data.with_columns(**{FILE_REF: None}))

        self.types: Final[ImageTypes] = types

    # - Methods
    def _where(self, col: CatalogColumn, value: str | Collection[str]) -> pl.DataFrame:
        series = self.data[col]
        predicate = series.is_in(value) if not isinstance(value, str) else series == value
        return self.data.filter(predicate)

    def where(
        self,
        col: CatalogColumn,
        value: EventType | ImageLike | Collection[EventType | ImageLike],
        inplace: bool = False,
    ) -> Catalog:
        df = self._where(col, value)
        return self._manager(df, inplace=inplace)

    def get_by_img_type(self, img_types: ImageLike | Collection[ImageLike], inplace=False) -> Catalog:
        df = self._where(IMG_TYPE, img_types)
        img_types = tuple((img_types,) if isinstance(img_types, str) else img_types)
        return self._manager(df, img_types=img_types, inplace=inplace)

    def get_by_event(self, event: EventType | Collection[EventType], inplace: bool = False) -> Catalog:
        return self.where(EVENT_TYPE, event, inplace=inplace)

    def intersect(self, x: ImageSequence, y: ImageSequence) -> tuple[Catalog, Catalog]:
        if not set(x).union(y) <= set(self.img_type):
            raise ValueError(
                f"Catalog does not contain all of the requested image types: {set(x).union(y) - set(self.img_type)}."
                " Use `Catalog.get_by_img_type` to get a subset of the catalog."
            )

        df_x, df_y = (self._where(IMG_TYPE, img_types) for img_types in (x, y))
        ids = set(np.intersect1d(df_x[ID].to_numpy(), df_y[ID].to_numpy()))  # type: set[str]
        df_x, df_y = (df.filter(df[ID].is_in(ids)) for df in (df_x, df_y))

        return (
            self._manager(df_x, inplace=False, img_types=x),
            self._manager(df_y, inplace=False, img_types=y),
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

    # =================================================================================================================
    # - Experimental
    def get_projection(self) -> pl.DataFrame:
        proj = self[PROJ].to_pandas().str.extract(PROJECTION_REGEX).astype({"lat": float, "lon": float, "a": float})
        return pl.from_pandas(proj)

    def with_projection(self) -> pl.DataFrame:
        from .plot import parse_projection

        return pl.concat(
            [self.data.drop([PROJ]), pl.DataFrame(self.data[PROJ].apply(parse_projection).to_list())],
            how="horizontal",
        )
