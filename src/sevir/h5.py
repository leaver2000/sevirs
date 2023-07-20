from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Final, Iterator, Literal, Mapping, TypeAlias, cast

import h5py
import numpy as np
import polars as pl
import tqdm
from numpy.typing import NDArray

from ._typing import AnyT, Array, N, Nd
from .constants import (
    DATA_INDEX,
    DEFAULT_FRAME_TIMES,
    FILE_INDEX,
    FILE_NAME,
    FLASH_TIME,
    FLASH_X,
    FLASH_Y,
    ID,
    IMG_TYPE,
    LGHT,
    ImageType,
)

if TYPE_CHECKING:
    # avoid potential circular import
    from .catalog import Catalog


def reshape_lightning_data(
    data: Array[Nd[N, Literal[5]], AnyT],
    *,
    img_size: int = 48,
    t_slice=slice(0, None),
    t_frame: Array[Nd[N], np.float64] = DEFAULT_FRAME_TIMES,
) -> Array[Nd[N, N, N, N], np.int16]:
    """Converts Nx5 lightning data matrix into a 2D grid of pixel counts
    this function was adopted from
    [eie-sevir](https://github.com/MIT-AI-Accelerator/eie-sevir/blob/master/sevir/generator.py#L386)

    >>> CONST_SIZE = 5
    >>> samples = 3 # rows
    >>> data = np.random.rand(samples, CONST_SIZE)
    >>> data.shape
    (3, 5)
    >>> data
    array([[0.31283714, 0.24021115, 0.81228598, 0.19904979, 0.58669168],
           [0.24984947, 0.96018605, 0.26226096, 0.86694522, 0.01303041],
           [0.5597265 , 0.26146486, 0.22932832, 0.18433348, 0.62103712]])
    >>> reshape_lightning_data(data).shape
    (1, 48, 48, 1)
    """
    shape = nx, ny, _ = (img_size, img_size, len(t_frame) if t_slice.stop is None else 1)

    if data.shape[0] == 0:  # there are no samples
        return np.zeros(shape, dtype=np.int16)

    # filter out points outside the grid
    x, y = data[:, FLASH_X], data[:, FLASH_Y]
    mask_xy = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    data = data[mask_xy, :]

    t = data[:, FLASH_TIME]
    # Filter/separate times
    if t_slice.stop is None:  # select only one time bin
        z = np.digitize(t, t_frame) - 1
        z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1
    else:  # compute z coordinate based on bin location times
        if t_slice.stop >= 0:  # special case:  frame 0 uses lght from frame 1
            mask_t = (t >= t_frame[0]) & (t < t_frame[1])  # np.logical_and(t >= t_frame[0], t < t_frame[1])
        elif t_slice.stop < len(t_frame):
            mask_t = (t >= t_frame[t_slice.stop - 1]) & (t < t_frame[t_slice.stop])
        else:
            mask_t = t >= t_frame[-1]
        data = data[mask_t, :]
        z = np.zeros(data.shape[0], dtype=np.int64)

    x, y = data[:, FLASH_X].astype(np.int64), data[:, FLASH_Y].astype(np.int64)
    lwt = np.ravel_multi_index([y, x, z], dims=shape)
    return np.bincount(lwt, minlength=np.prod(shape)).reshape((1,) + shape).astype(np.int16)


class H5File(h5py.File):
    """subclass of h5py.File that provides a few convenience methods for SEVIR files and __reduce__ to allow
    for pickling which is required for multiprocessing."""

    __slots__ = ("image_type",)
    if TYPE_CHECKING:
        image_type: Final[ImageType]

    def __init__(self, filename: str, image_type: ImageType) -> None:
        super().__init__(filename, mode="r")
        self.image_type = image_type

    @property
    def event_ids(self) -> NDArray[np.bytes_]:
        return super().__getitem__(ID).__getitem__()[...]  # type: ignore[unused-ignore]

    def __getitem__(self, __id: bytes | str, /) -> Array[Nd[N, N, N, N], np.int16]:
        img_t = self.image_type

        return cast(
            Array[Nd[N, N, N, N], np.int16],
            reshape_lightning_data(super().__getitem__(__id)) if img_t == LGHT else super().__getitem__(img_t),  # type: ignore[unused-ignore] # noqa: E501
        )

    def get_by_file_index(self, index: int) -> Array[Nd[N, N, N, N], np.int16]:
        return self[self.event_ids[index]]


ImageGroupIterator: TypeAlias = Iterator[tuple[tuple[str, ImageType], H5File]]


class H5Store(Mapping[str | bytes, list[Array[Nd[N, N, N, N], np.int16]]]):
    __slots__ = ("_data", "_dfdx", "_dbar", "_dkey")
    if TYPE_CHECKING:
        _dfdx: pl.DataFrame
        _data: list[H5File]
        _dbar: tqdm.tqdm
        _dkey: set[bytes]

    # - Initialization
    def __init__(self, catalog: Catalog, nproc: int | None = None) -> None:
        # validate the catalog before attempting to open files
        required_cols = [FILE_NAME]
        index_cols = [ID, IMG_TYPE, FILE_INDEX]
        required_cols += index_cols

        self._dfdx = df = catalog.to_polars().with_columns(**{DATA_INDEX: None})
        if not set(df.columns).issuperset(required_cols):
            raise ValueError(f"Catalog is missing columns: {required_cols}")

        logging.info(f"Loading {df[FILE_NAME].n_unique()} files with {df[IMG_TYPE].n_unique()} image types.")
        self._data = []
        self._dkey = set(df[ID])
        self._dbar = tqdm.tqdm(total=df[FILE_NAME].n_unique())

        for (f_name, img_t), _ in self.iter_groups():
            self._dfdx = df = self._dfdx.with_columns(
                pl.when(df[FILE_NAME] == f_name).then(len(self._data)).otherwise(df[DATA_INDEX]).alias(DATA_INDEX)
            )
            self._data.append(H5File(f_name, img_t))
            self._dbar.update(1)
        # TODO: need to create a Thread Safe object to store the file ref in.
        # if nproc is None:
        #
        # else:
        #     with multiprocessing.pool.ThreadPool(nproc) as pool:
        #         pool.starmap(self._load_dataframe_index, self.iter_groups())

        self._dbar.close()

    def _load_dataframe_index(self, group: tuple[str, ImageType], df: pl.DataFrame) -> None:
        """
        instead of putting the files into the dataframe, a reference to the index of the file in the list is stored
        in the DataFrame Index `dfdx`
        """
        f_name, img_t = group
        self._dfdx = df.with_columns(
            pl.when(df[FILE_NAME] == f_name).then(len(self._data)).otherwise(df[DATA_INDEX]).alias(DATA_INDEX)
        )
        self._data.append(H5File(f_name, img_t))
        self._dbar.update(1)

    def iter_groups(self) -> ImageGroupIterator:
        columns = pl.col(FILE_NAME), pl.col(IMG_TYPE), pl.col(DATA_INDEX)
        return cast(ImageGroupIterator, iter(self.index.select(columns).groupby(by=[FILE_NAME, IMG_TYPE])))

    # - Properties
    @property
    def index(self) -> pl.DataFrame:
        return self._dfdx

    @property
    def data(self) -> list[H5File]:
        return copy.copy(self._data)

    # - Mapping interface
    def __getitem__(
        self, __idx: str | bytes | tuple[str | bytes, list[ImageType]]
    ) -> list[Array[Nd[N, N, N, N], np.int16]]:
        if isinstance(__idx, tuple):
            __idx, image_ids = __idx
        else:
            image_ids = None
        if isinstance(__idx, bytes):
            __idx = __idx.decode("utf-8")

        di_fi = self.index.filter(
            self.index[ID] == __idx
            if image_ids is None
            else (self.index[ID] == __idx) & (self.index[IMG_TYPE].is_in(image_ids))
        )
        return [
            self._data[di][__idx][fi : fi + 1, :, :, :]
            for di, fi in di_fi.select(pl.col(DATA_INDEX), pl.col(FILE_INDEX)).iter_rows()
        ]

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._dkey)

    def __len__(self) -> int:
        return len(self._data)

    # - Methods
    def close(self) -> None:
        for f in self._data:
            f.close()
        self._data.clear()
        self._dfdx = self._dfdx.with_columns(**{DATA_INDEX: None})
