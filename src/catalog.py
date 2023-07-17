from __future__ import annotations

import functools
import os
import typing

import pandas as pd
import xarray as xr
from pandas._libs.missing import NAType

from ._typing import ColumnIndexerType, LocIndexerType, Self
from numpy.typing import NDArray
from .constants import (
    CATALOG_BOUNDING_BOX,
    CATALOG_DTYPES,
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_IMAGE_TYPES,
    EVENT_TYPE,
    FILE_INDEX,
    FILE_NAME,
    HEIGHT_M,
    ID,
    IMG_TYPE,
    PATH_TO_SEVIR,
    PROJ,
    TIME_UTC,
    WIDTH_M,
    EventType,
    ImageType,
    ImageTypeSet,
)

sel = pd.IndexSlice
P = typing.ParamSpec("P")


def html(obj: typing.Any) -> str:
    if not hasattr(obj, "_repr_html_"):
        return repr(obj)
    return obj._repr_html_()


def set_and_sort_index(func: typing.Callable[P, pd.DataFrame]) -> typing.Callable[P, pd.DataFrame]:
    index_names = [ID, IMG_TYPE]

    @functools.wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
        df = func(*args, **kwargs)

        return (
            df
            if df.index.names == index_names
            else df.set_index(index_names, drop=True, append=False).sort_index(axis=0, level=ID, inplace=False)
        )

    return decorator


@set_and_sort_index
def read(
    __src: str,
    /,
    *,
    img_types: ImageTypeSet | None = None,
    drop: list[str] = [PROJ, HEIGHT_M, WIDTH_M],
) -> pd.DataFrame:
    """
    if no image types are specified the img_types are not filtered so there will be storm ids that dont have all of
    the image types.

    >>> import sevir
    >>> df = sevir.catalog.read(sevir.DEFAULT_CATALOG)
    >>> df.loc[pd.IndexSlice[0:10:2, [ImageTypes, ...]], [columns, ...]]
    """
    df = pd.read_csv(__src, parse_dates=[TIME_UTC], low_memory=False, dtype=CATALOG_DTYPES).drop(columns=drop)
    # remove all rows that don't have the selected image types and set the index to the ID column
    # group all of the files by their ID, and remove any where there are not complete set of image types
    return subset_by_image_types(df, img_types) if img_types is not None else df


@set_and_sort_index
def subset_by_image_types(df: pd.DataFrame | Catalog, img_types: ImageTypeSet) -> pd.DataFrame:
    """Subset the catalog to only include the image types specified in img_types"""
    if isinstance(df, Catalog):
        df = df.to_pandas()
    elif not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=False)

    # filter the rows that dont contain the selected image types
    df = df.loc[df[IMG_TYPE].isin(img_types)].set_index(ID)
    # create a mask to filter out the IDs that dont have all of the image types
    mask = df.groupby(ID)[IMG_TYPE].size() >= len(img_types)
    # the mask is a different shape because of the groupby, so we need to use the index to filter the dataframe
    return df.loc[mask.index[mask], :].reset_index(drop=False)
    # return df.set_index([ID, IMG_TYPE], drop=True, append=False).sort_index(axis=0, level=ID)


def resolve(left: str | None, right: str | None = None) -> str:
    """resolve two strings to a single path.  If both are provided, they are joined."""
    if left is not None and right is not None:
        out = os.path.join(left, right)
        assert os.path.exists(out), f"Path does not exist: {out}"
        return out
    elif right is not None:
        assert os.path.exists(right), f"Path does not exist: {right}"
        return right
    elif left is not None:
        assert os.path.exists(left), f"Path does not exist: {left}"
        return left

    raise ValueError("There is no path to resolve")


class Catalog:
    """This implementation of the SEVIR Catalog uses a MultiIndex to allow for grouping and slicing events by the
    event_index and image_type.  The __getitem__ method of the catalog is slightly different than than a typical pandas
    DataFrame as it prioritizes the index over the columns.


    ```python
    >>> import sevir
    >>> cat = sevir.Catalog()
    >>> cat
                                       id                                          file_name  file_index  ...   data_min       data_max  pct_missing
    event_index img_type                                                                                  ...
    0           ir069     R18032123577290  /mnt/nuc/c/sevir/data/ir069/2018/SEVIR_IR069_R...         166  ... -63.520329     -33.208508     0.074978
                ir107     R18032123577290  /mnt/nuc/c/sevir/data/ir107/2018/SEVIR_IR107_R...         166  ... -61.731163       7.948836     0.074978
                lght      R18032123577290  /mnt/nuc/c/sevir/data/lght/2018/SEVIR_LGHT_ALL...           0  ...   0.000000   78711.000000     0.000000
                vil       R18032123577290  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         760  ...   0.000000     188.000000     0.000000
                vis       R18032123577290  /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...         129  ...   0.014424       0.803247     0.071392
    ...                               ...                                                ...         ...  ...        ...            ...          ...
    12636       ir069             S858968  /mnt/nuc/c/sevir/data/ir069/2019/SEVIR_IR069_S...         544  ... -68.739433     -23.089878     0.000000
                ir107             S858968  /mnt/nuc/c/sevir/data/ir107/2019/SEVIR_IR107_S...         543  ... -68.275558      20.554308     0.000000
                lght              S858968  /mnt/nuc/c/sevir/data/lght/2019/SEVIR_LGHT_ALL...           0  ...   0.000000  453187.000000     0.000000
                vil               S858968  /mnt/nuc/c/sevir/data/vil/2019/SEVIR_VIL_STORM...         421  ...   0.000000     254.000000     0.000000
                vis               S858968  /mnt/nuc/c/sevir/data/vis/2019/SEVIR_VIS_STORM...          75  ...   0.033034       1.115368     0.000000

    [63185 rows x 19 columns]

    >>> import pandas as pd
    >>> idx = pd.IndexSlice
    >>> cat[idx[0:5, "vil"], :]
                                       id                                          file_name  file_index  ... data_min data_max  pct_missing
    event_index img_type                                                                                  ...
    0           vil       R18032123577290  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         760  ...      0.0    188.0          0.0
    1           vil       R18032123577314  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         759  ...      0.0    254.0          0.0
    2           vil       R18032123577327  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         762  ...      0.0    254.0          0.0
    3           vil       R18032123577328  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         763  ...      0.0    254.0          0.0
    4           vil       R18032123577354  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         761  ...      0.0    254.0          0.0
    5           vil       R18032123577787  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         757  ...      0.0    190.0          0.0

    [6 rows x 19 columns]
    >>> cat[idx[0:25:5, ['vis', 'vil']], 'file_name']
    <class 'tuple'>
                                                                  file_name
    event_index img_type
    0           vil       /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...
                vis       /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...
    5           vil       /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...
                vis       /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...
    10          vil       /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...
                vis       /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...
    15          vil       /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...
                vis       /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...
    20          vil       /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...
                vis       /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...
    25          vil       /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...
                vis       /mnt/nuc/c/sevir/data/vis/2018/SEVIR_VIS_RANDO...
    ```
    """

    # ==========================================================================
    # Constructors
    def _manager(self, df: pd.DataFrame, inplace: bool) -> Self:
        if inplace:
            self._data = df
            return self
        return self.__class__(df)

    def __init__(
        self,
        __value: Catalog | pd.DataFrame | str = PATH_TO_SEVIR,
        *,
        catalog: str = DEFAULT_CATALOG,
        data_dir: str = DEFAULT_DATA,
        img_types: ImageTypeSet = DEFAULT_IMAGE_TYPES,
        validate: bool = False,
    ) -> None:
        if isinstance(__value, str):
            df = read(resolve(__value, catalog), img_types=img_types).sort_index(axis=0, level=[ID, IMG_TYPE])
            if catalog or data_dir:
                prefix = resolve(__value, data_dir)
                df[FILE_NAME] = df[FILE_NAME].map(lambda s: os.path.join(prefix, s)).astype("string")
        elif isinstance(__value, Catalog):
            df = __value.to_pandas()
        elif isinstance(__value, pd.DataFrame):
            df = __value
        else:
            raise TypeError(f"catalog must be Catalog, pd.DataFrame, or str, not {type(__value)}")

        self._data = df

        if validate:
            self.validate()

    # =================================================================================================================
    # Properties
    @property
    def columns(self) -> pd.Index:
        return self._data.columns

    @property
    def index(self) -> pd.MultiIndex:
        return self._data.index  # type: ignore

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def dtypes(self) -> pd.Series[str]:
        return self._data.dtypes

    @property
    def img_types(self) -> set[ImageType]:
        return set(ImageType.map(self.get_level_values(IMG_TYPE)))

    @property
    def file_name(self) -> pd.Series[str]:
        return self._data[FILE_NAME]

    @property
    def bbox(self) -> pd.DataFrame:
        return self._data[CATALOG_BOUNDING_BOX]

    @property
    def time_utc(self) -> pd.Series[pd.Timestamp]:
        return self._data[TIME_UTC]

    @property
    def event_type(self) -> pd.Series[EventType | NAType]:  # type: ignore
        return self._data[EVENT_TYPE]

    @property
    def file_index(self) -> pd.Series[int]:
        return self._data[FILE_INDEX]

    # =================================================================================================================
    def __getitem__(self, idx: LocIndexerType[typing.Any]) -> pd.DataFrame:
        """I generic getitem method that returns the pandas DataFrame"""
        # because the index is a multi index the type checker does not understand that a single
        # integer will still return a pandas DataFrame and not a Series
        obj = self._data.loc.__getitem__(idx)
        return obj.to_frame() if isinstance(obj, pd.Series) else obj

    def __repr__(self) -> str:
        return repr(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _repr_html_(self) -> str:
        return html(self._data)

    # =================================================================================================================
    def to_pandas(self, columns: list[str] | pd.Index | pd.MultiIndex | None = None) -> pd.DataFrame:
        """copy the data to a pandas DataFrame to that the original object is not modified."""
        df = self._data.copy()
        if columns is not None:
            df = df[columns]
        return df

    def to_xarray(self, columns) -> xr.Dataset:
        return xr.Dataset.from_dataframe(self.to_pandas(columns))

    def to_numpy(self, columns) -> NDArray:
        return self.to_pandas(columns).to_numpy()

    def get_level_values(self, level: str | int) -> pd.Index:
        return self._data.index.get_level_values(level)

    def validate(self) -> Catalog:
        if self._data.empty:
            raise ValueError("Catalog is empty.")
        for file in self.file_name:
            if not os.path.exists(file):
                raise FileNotFoundError(file)
        return self

    def get_by_img_type(
        self, img_types: list[ImageType], columns: ColumnIndexerType[str] = slice(None), inplace=False
    ) -> Catalog:
        df = self[sel[:, img_types], columns]
        return self._manager(df, inplace)

    def get_by_event(self, event: EventType, inplace: bool = False) -> Catalog:
        df = self[self.event_type == event, :]
        return self._manager(df, inplace)

    def split_by_types(self, x: list[ImageType], y: list[ImageType]) -> tuple[Catalog, Catalog]:
        if len(self.img_types) != len(x + y) != len(set(x + y)):
            raise ValueError(
                f"Catalog does not contain all of the requested image types: {set(x + y) - self.img_types}."
                " Use `Catalog.get_by_img_type` to get a subset of the catalog."
            )

        return self.get_by_img_type(x), self.get_by_img_type(y)


def main() -> None:
    cat = Catalog(
        # by default with no arguments the catalog will be read from the default location
    )
    assert isinstance(cat, Catalog)
    assert not isinstance(cat, pd.DataFrame)
    assert isinstance(cat.columns, pd.Index)
    assert isinstance(cat.index, pd.MultiIndex)
    print(cat._data.loc[sel["R18032123577290", :, :], :])
    x, y = cat.split_by_types(
        [ImageType.IR_069, ImageType.VISIBLE, ImageType.IR_107],
        [ImageType.VERTICALLY_INTEGRATED_LIQUID, ImageType.LIGHTNING],
    )
    print(x, y, sep="\n")

    # validate all the paths
    assert cat is cat.validate()
    assert cat is not Catalog(cat)
    assert cat is cat.get_by_event("Tornado", inplace=True)
    assert cat is not cat.get_by_event("Tornado", inplace=False)
    cat.to_pandas()


if __name__ == "__main__":
    main()
