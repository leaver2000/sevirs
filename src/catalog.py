from __future__ import annotations

import logging
import os
import typing
from typing import Any, Callable

import pandas as pd

try:
    from tqdm import tqdm as RW
except ImportError:
    logging.info("You need to install tqdm to use progress bar")
    RW = list
if typing.TYPE_CHECKING:
    from pandas._typing import HashableT, IndexType, MaskType, Scalar
    from pandas.core.indexing import _IndexSliceTuple as IndexSliceTuple
else:
    HashableT = typing.TypeVar("HashableT", bound=typing.Hashable)
    Scalar = Any
    IndexType = Any
    MaskType = Any
    IndexSliceTuple = Any

from .constants import (
    CATALOG_DTYPES,
    DATA_MAX,
    DATA_MIN,
    DEFAULT_CATALOG,
    DEFAULT_DATA_HOME,
    DEFAULT_FRAME_TIMES,
    DEFAULT_N_FRAMES,
    EPISODE_ID,
    EVENT_ID,
    EVENT_INDEX,
    EVENT_TYPE,
    FILE_INDEX,
    FILE_NAME,
    HEIGHT_M,
    ID,
    IMG_TYPE,
    IR_069,
    IR_107,
    LIGHTNING,
    LL_LAT,
    LL_LON,
    MINUTE_OFFSETS,
    PATH_TO_SEVIR,
    PCT_MISSING,
    PROJ,
    SEVIR_DTYPES,
    SIZE_X,
    SIZE_Y,
    TIME_UTC,
    UR_LAT,
    UR_LON,
    VERTICALLY_INTEGRATED_LIQUID,
    VISIBLE,
    WIDTH_M,
    SEVIRImageType,
)

ColumnIndexer: typing.TypeAlias = """(
    slice
    | HashableT
    | IndexType
    | MaskType
    | Callable[[pd.DataFrame], IndexType | MaskType | list[HashableT]]
    | list[HashableT]
)
"""
LocIndexer: typing.TypeAlias = """(
    int
    | ColumnIndexer[HashableT]
    | tuple[
        IndexType
        | MaskType 
        | list[HashableT] 
        | slice 
        | IndexSliceTuple 
        | Callable, list[HashableT] 
        | slice 
        | pd.Series[bool] 
        | Callable
    ]
)"""


def html(obj) -> str:
    if not hasattr(obj, "_repr_html_"):
        return repr(obj)
    return obj._repr_html_()


class SEVIRBase:
    __slots__ = ("_values", "_index")
    if typing.TYPE_CHECKING:
        _values: pd.DataFrame
        _index: pd.MultiIndex

    def __init__(self, values: pd.DataFrame) -> None:
        self._values = values
        self._index = typing.cast(pd.MultiIndex, values.index)

    @property
    def values(self) -> pd.DataFrame:
        return self._values.copy()

    @property
    def index(self) -> pd.MultiIndex:
        return self._index.copy()

    def __getitem__(self, idx: LocIndexer[typing.Any]) -> pd.DataFrame:
        """I generic getitem method that returns the pandas DataFrame"""
        # because the index is a multi index the type checker does not understand that a single
        # integer will still return a pandas DataFrame and not a Series
        obj = self._values.loc.__getitem__(idx)
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        return obj

    def __repr__(self) -> str:
        return repr(self._values)

    def _repr_html_(self) -> str:
        return html(self._values)


class SEVIRCatalog(SEVIRBase):

    """This implementation of the SEVIR Catalog uses a MultiIndex to allow for grouping and slicing events by the
    event_index and image_type.  The __getitem__ method of the catalog is slightly different than than a typical pandas
    DataFrame as it prioritizes the index over the columns.


    ```python
    >>> from sevir.catalog import SEVIRCatalog
    >>> cat = SEVIRCatalog()
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
    >>> ixs = pd.IndexSlice
    >>> cat[ixs[0:5, "vil"], :]
                                       id                                          file_name  file_index  ... data_min data_max  pct_missing
    event_index img_type                                                                                  ...
    0           vil       R18032123577290  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         760  ...      0.0    188.0          0.0
    1           vil       R18032123577314  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         759  ...      0.0    254.0          0.0
    2           vil       R18032123577327  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         762  ...      0.0    254.0          0.0
    3           vil       R18032123577328  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         763  ...      0.0    254.0          0.0
    4           vil       R18032123577354  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         761  ...      0.0    254.0          0.0
    5           vil       R18032123577787  /mnt/nuc/c/sevir/data/vil/2018/SEVIR_VIL_RANDO...         757  ...      0.0    190.0          0.0

    [6 rows x 19 columns]
    >>> cat[ixs[0:25:5, ['vis', 'vil']], 'file_name']
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

    def __init__(
        self,
        catalog: str | pd.DataFrame = DEFAULT_CATALOG,
        *,
        prefix: str | None = DEFAULT_DATA_HOME,
        image_types: set[SEVIRImageType] = set(SEVIRImageType),
        shuffle: int | None = None,
    ) -> None:
        if isinstance(catalog, pd.DataFrame):
            df = catalog
        else:
            df = self._read_catalog(catalog, image_types)
        if shuffle:
            df = df.sample(frac=1)
        if prefix:
            df[FILE_NAME] = [os.path.join(prefix, f) for f in df[FILE_NAME]]

        super().__init__(df.sort_index(axis=0, level=EVENT_INDEX))

    def _read_catalog(self, catalog: str, image_types: set[SEVIRImageType]) -> pd.DataFrame:
        df = (
            pd.read_csv(
                catalog,
                parse_dates=[TIME_UTC],
                low_memory=False,
                dtype=CATALOG_DTYPES,
            )
            .drop(columns=[PROJ])
            .drop_duplicates()
        )
        # remove all rows that don't have the selected image types
        df = df.loc[df[IMG_TYPE].isin(image_types)]
        # the ID columns is a string with either a "S" or "R" prefix.
        # df[EVENT_INDEX] = df[ID].str.slice(1).astype(int)
        # set the index to the ID column
        df.set_index([ID], inplace=True)
        # group all of the files by their ID, and remove any where there are not complete set of image types
        mask = df.groupby(ID)[IMG_TYPE].size() == len(set(image_types))
        # mask out the index to only include the IDs that have all of the image types
        df = df.loc[mask[mask].index, :]
        ids = df.index.get_level_values(ID).to_frame(index=False)
        df[EVENT_INDEX] = ids.assign(event_index=ids.groupby(ID).ngroup()).event_index.to_numpy()

        return df.reset_index().set_index([EVENT_INDEX, IMG_TYPE])

    @property
    def file_names(self) -> pd.Series[str]:
        return self._values[FILE_NAME]

    def validate_paths(self) -> SEVIRCatalog:
        for file in self.get_paths():
            if not os.path.exists(file):
                raise FileNotFoundError(file)
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self.values

    def get_paths(self) -> pd.Series[str]:
        return self._values[FILE_NAME]

    def select_by_types(
        self, image_types: list[SEVIRImageType], columns: ColumnIndexer[str] = slice(None)
    ) -> SEVIRCatalog:
        return SEVIRCatalog(self[pd.IndexSlice[:, list(image_types)], columns].copy(), image_types=set(image_types))

    def split_by_types(self, x: list[SEVIRImageType], y: list[SEVIRImageType]) -> tuple[SEVIRCatalog, SEVIRCatalog]:
        return self.select_by_types(x), self.select_by_types(y)


def main():
    print(SEVIRCatalog())


if __name__ == "__main__":
    main()
