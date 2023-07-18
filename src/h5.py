from __future__ import annotations

import logging
import multiprocessing.pool
import typing

import h5py
import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray
from pandas.core.groupby.generic import SeriesGroupBy
from torch.utils.data import Dataset

from ._typing import AnyT, Array, N, Nd
from .catalog import Catalog
from .constants import DEFAULT_FRAME_TIMES as FRAME_TIMES
from .constants import (
    FILE_INDEX,
    FILE_NAME,
    FLASH_TIME,
    FLASH_X,
    FLASH_Y,
    ID,
    IMG_TYPE,
    LIGHTNING,
    ImageType,
)


def reshape_lightning_data(
    data: Array[Nd[N, typing.Literal[5]], AnyT],  # NumpyArray[Nd[N, typing.Literal[5]]],
    *,
    time_slice=slice(0, None),
    img_size: int = 48,
    dtype: type[AnyT] = np.int16,
) -> Array[Nd[N, N, N, N], AnyT]:
    """Converts Nx5 lightning data matrix into a 2D grid of pixel counts

    >>> CONST_SIZE = 5
    >>> samples = 3 # rows
    >>> data = np.random.rand(samples, CONST_SIZE)
    >>> data.shape
    (3, 5)
    >>> data
    array([[0.31283714, 0.24021115, 0.81228598, 0.19904979, 0.58669168],
           [0.24984947, 0.96018605, 0.26226096, 0.86694522, 0.01303041],
           [0.5597265 , 0.26146486, 0.22932832, 0.18433348, 0.62103712]])
    >>> _lght_to_grid_new(data).shape
    (1, 48, 48, 1)

    """
    shape = n_length, n_width, _ = (img_size, img_size, len(FRAME_TIMES) if time_slice.stop is None else 1)

    if data.shape[0] == 0:  # there are no samples
        return np.zeros(shape, dtype=dtype)

    # filter out points outside the grid
    x, y = data[:, FLASH_X], data[:, FLASH_Y]

    mask_x_y = np.logical_and.reduce([x >= 0, x < n_length, y >= 0, y < n_width])
    data = data[mask_x_y, :]

    t = data[:, FLASH_TIME]
    # Filter/separate times
    if time_slice.stop is None:  # select only one time bin
        z = np.digitize(t, FRAME_TIMES) - 1
        z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1
    else:  # compute z coordinate based on bin location times
        if time_slice.stop >= 0:  # special case:  frame 0 uses lght from frame 1
            mask_time = np.logical_and(t >= FRAME_TIMES[0], t < FRAME_TIMES[1])
        elif time_slice.stop < len(FRAME_TIMES):
            mask_time = np.logical_and(t >= FRAME_TIMES[time_slice.stop - 1], t < FRAME_TIMES[time_slice.stop])
        else:
            mask_time = t >= FRAME_TIMES[-1]
        data = data[mask_time, :]

        z = np.zeros(data.shape[0], dtype=np.int64)
    x, y = data[:, FLASH_X].astype(np.int64), data[:, FLASH_Y].astype(np.int64)
    lwt = np.ravel_multi_index([y, x, z], dims=shape)  # type: ignore
    return np.bincount(lwt, minlength=np.prod(shape)).reshape((1,) + shape).astype(dtype)


class H5File(h5py.File):
    """subclass of h5py.File that provides a few convenience methods for SEVIR files and __reduce__ to allow
    for pickling which is required for multiprocessing."""

    __slots__ = ("_img_type",)
    if typing.TYPE_CHECKING:
        _img_type: ImageType

    def __init__(self, filename: str, img_type: str) -> None:
        super().__init__(filename, mode="r")
        self._img_type = ImageType(img_type)

    @property
    def img_type(self) -> ImageType:
        return self._img_type

    @property
    def event_ids(self) -> NDArray[np.bytes_]:
        return super().__getitem__(ID)[...]  # type: ignore

    def __getitem__(self, __id: bytes | str, /) -> Array[Nd[N, N, N, N], np.int16]:
        img_t = self._img_type

        return typing.cast(
            Array[Nd[N, N, N, N], np.int16],
            reshape_lightning_data(super().__getitem__(__id))  # type: ignore
            if img_t == LIGHTNING
            else super().__getitem__(img_t),  # type: ignore
        )

    def get_by_file_index(self, index: int) -> Array[Nd[N, N, N, N], np.int16]:
        return self[self.event_ids[index]]


DATA_INDEX = "data_index"


class H5Store(typing.Mapping[str | bytes, list[Array[Nd[N, N, N, N], np.int16]]]):
    __slots__ = ("_data", "_sidx", "_dbar", "_keys")
    if typing.TYPE_CHECKING:
        _sidx: pd.Series[int]
        _data: list[H5File]
        _dbar: tqdm.tqdm

    def __init__(self, catalog: Catalog, nproc: int | None = 1) -> None:
        # validate the catalog before attempting to open files
        index_cols = [ID, IMG_TYPE, FILE_INDEX]
        required_cols = [FILE_NAME] + index_cols
        df = catalog.to_pandas().reset_index(drop=False).astype({ID: "bytes"})
        if not set(df.columns).issuperset(required_cols):
            raise ValueError(f"Catalog is missing columns: {required_cols}")

        self._sidx = pd.Series(index=pd.MultiIndex.from_frame(df[index_cols]), name=DATA_INDEX, dtype="Int64")
        self._keys = set(df[ID])
        self._data = []

        logging.info("Opening HDF5 files")
        self._dbar = tqdm.tqdm(total=df[FILE_NAME].nunique())
        with multiprocessing.pool.ThreadPool(nproc) as pool:
            pool.starmap(
                self._load_file_series, df[[FILE_NAME, ID, IMG_TYPE, FILE_INDEX]].groupby(by=[FILE_NAME, IMG_TYPE])
            )

    def _load_file_series(self, file_nt: tuple[str, str], df: pd.DataFrame) -> None:
        self._sidx.loc[df[[ID, IMG_TYPE, FILE_INDEX]].to_numpy()] = len(self._data)
        self._data.append(H5File(*file_nt))
        self._dbar.update(1)

    # =========================================================================
    # Mapping interface

    def __getitem__(self, __id: str | bytes) -> list[Array[Nd[N, N, N, N], np.int16]]:
        if isinstance(__id, str):
            __id = __id.encode("utf-8")
        df = self._sidx.loc[__id, :, :].reset_index()
        data_file_index = typing.cast(typing.Iterable[tuple[int, int]], df[[DATA_INDEX, FILE_INDEX]].to_numpy())
        return [self._data[di][__id][fi : fi + 1, :, :, :] for di, fi in data_file_index]

    def __iter__(self) -> typing.Iterator[bytes]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._data)


def main() -> None:
    from .catalog import Catalog

    # "vis", "ir069", "ir107", "vil", "lght"
    store = H5Store(Catalog(img_types=("vis", "ir069", "ir107", "vil", "lght")))
    d = store["R18032123577290"]
    for a in d:
        print(a.shape)
    # print(l.shape, r.shape)
    #     np.array(store["R18032123577290"])
    #     )


if __name__ == "__main__":
    main()
