from __future__ import annotations

import logging
import multiprocessing.pool
import os
import typing

import h5py
import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray
from pandas.core.groupby.generic import SeriesGroupBy
from typing_extensions import TypeVarTuple, Unpack

from ._typing import AnyT, N, Nd, Scalar
from .abc import MappedDataset
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
    data: np.ndarray[Nd[N, typing.Literal[5]], AnyT],  # NumpyArray[Nd[N, typing.Literal[5]]],
    *,
    time_slice=slice(0, None),
    img_size: int = 48,
    dtype: type[AnyT] = np.int16,
) -> np.ndarray[Nd[N, N, N], AnyT]:
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
        return np.zeros(shape, dtype=dtype)  # type: ignore

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
    # x, y = data[:, RASTER_X].astype(np.int64), data[:, RASTER_Y].astype(np.int64)
    lwt = np.ravel_multi_index([y, x, z], dims=shape)
    return np.bincount(lwt, minlength=np.prod(shape)).reshape(shape).astype(dtype)  # type: ignore


class H5File(h5py.File, typing.Generic[AnyT]):
    """subclass of h5py.File that provides a few convenience methods for SEVIR files and __reduce__ to allow
    for pickling which is required for multiprocessing."""

    __slots__ = ("_img_type", "_metadata")
    if typing.TYPE_CHECKING:
        _img_type: ImageType
        _metadata: AnyT | None
        __getitem__: typing.Callable[[str | bytes], h5py.Dataset]

    def __init__(self, filename: str, img_type: str | None = None, *, metadata: AnyT = None) -> None:
        super().__init__(filename, mode="r")
        self._img_type = ImageType(img_type)
        self._metadata = metadata

    @property
    def img_type(self) -> ImageType:
        return self._img_type

    @property
    def metadata(self) -> AnyT:
        return self._metadata  # type: ignore

    @property
    def event_ids(self) -> NDArray[np.bytes_]:
        return self[ID][...]

    def get_by_event_id(self, __id: bytes | str, /) -> np.ndarray[Nd[N, N, N, N], np.dtype[np.int16]]:
        img_t = self._img_type
        arr = (
            reshape_lightning_data(self[__id][...], dtype=np.int16)
            if img_t == LIGHTNING
            else self[img_t][self[ID] == __id]
        )
        return arr[np.newaxis, ...]  # type: ignore

    def get_by_file_index(self, index: int) -> np.ndarray[Nd[N, N, N, N], np.dtype[np.int16]]:
        return self.get_by_event_id(self.event_ids[index])

    def get_by(self, by: bytes | str | NDArray[np.bytes_]) -> np.ndarray[Nd[N, N, N, N], np.dtype[np.int16]]:
        ...


Ts = TypeVarTuple("Ts", default=Unpack[tuple[Scalar, ...]])


class H5StoreBase(MappedDataset[tuple[Unpack[Ts]], H5File[pd.DataFrame]]):
    """The h5py.File object does not like being inserted into a pandas DataFrame or a numpy array. This class
    provides a wrapper around h5py.File that allows for this. It also provides a convenient way to index into
    the file using a pandas DataFrame."""

    __slots__ = ("_series", "_files", "_loc")
    if typing.TYPE_CHECKING:
        from pandas.core.series import _LocIndexerSeries

        _series: pd.Series[int]
        _files: list[H5File[pd.DataFrame]]
        _loc: _LocIndexerSeries[int]

    def __init__(self, index: pd.MultiIndex) -> None:
        self._series = s = pd.Series(index=index, name=self.index_name, dtype="Int64")
        self._loc = s.loc
        self._files = []

    def __setitem__(self, key: np.ndarray, file: H5File) -> None:
        self._series[key] = len(self._files)
        self._files.append(file)

    def __getitem__(self, index: tuple[Unpack[Ts]]) -> H5File[pd.DataFrame]:
        """

        The loader stores the Files in a dictionary with keys mapped to a (ID, ImageType) tuple.
        >>> key = ("S728503", "vis") # (ID, ImageType)
        >>> nlwt = pd.IndexSlice[0:1, :, :, 0] # (N, L, W, T)
        >>> loader.dataset[key].loc[nlwt] # type: numpy.ndarray


        All raster types in SEVIR (`vis`, `ir069`, `ir107` & `vil`) store the data 4D tensors with shapes `N x L x W x T`,
        where `N` is the number of events in the file,  `LxL` is the image size (see patch size column in Table 1),
        and `T` is the number of time steps in the image sequence.

        >>> loader = SEVIRReaderHDF5(...)
        >>> nlwt = pd.IndexSlice[0:1, :, :, 0] # (N, L, W, T)
        >>> loader[("S728503", "vis"), nlwt]
        """
        # The series is a MultiIndex Store mapped to the file index.
        idx = self.loc.__getitem__(index)  # type: ignore
        return self._files.__getitem__(idx)

    @property
    def index(self) -> pd.MultiIndex:
        return self._series.index  # type: ignore

    @property
    def index_name(self) -> str:
        return f"<{self.__class__.__name__}.index>"

    @property
    def loc(self) -> _LocIndexerSeries[int]:
        return self._loc

    def invert_index(self) -> pd.DataFrame:
        return self._series.reset_index(inplace=False).set_index(self.index_name, inplace=False)

    def close_all(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()
        self._series = self._series.iloc[0:0]


class H5Store(H5StoreBase[str, str, int]):
    __slots__ = ("data",)

    def __init__(self, catalog: Catalog, nproc: int | None = os.cpu_count()) -> None:
        # validate the catalog before attempting to open files
        index_cols = [ID, IMG_TYPE, FILE_INDEX]
        required_cols = [FILE_NAME] + index_cols  # @#[FILE_NAME, ID, IMG_TYPE, FILE_INDEX]
        df = catalog.validate().to_pandas().reset_index(drop=False)
        if not set(df.columns).issuperset(required_cols):
            raise ValueError(f"Catalog is missing columns: {required_cols}")

        super().__init__(pd.MultiIndex.from_frame(df[index_cols]))

        logging.info("Opening HDF5 files")
        with multiprocessing.pool.ThreadPool(nproc) as pool:
            bar = tqdm.tqdm(total=df.file_name.nunique())
            for file in pool.starmap(
                lambda filename_img_t, df: H5File[pd.DataFrame](*filename_img_t, metadata=df),
                df[[FILE_NAME, ID, IMG_TYPE, FILE_INDEX]].groupby(by=[FILE_NAME, IMG_TYPE]),
            ):
                self[file.metadata[index_cols].to_numpy()] = file
                bar.update(1)

    def __getitem__(self, index: tuple[str, str, int]) -> H5File[pd.DataFrame]:
        """

        The loader stores the Files in a dictionary with keys mapped to a (ID, ImageType) tuple.
        >>> key = ("S728503", "vis") # (ID, ImageType)
        >>> nlwt = pd.IndexSlice[0:1, :, :, 0] # (N, L, W, T)
        >>> loader.dataset[key].loc[nlwt] # type: numpy.ndarray


        All raster types in SEVIR (`vis`, `ir069`, `ir107` & `vil`) store the data 4D tensors with shapes `N x L x W x T`,
        where `N` is the number of events in the file,  `LxL` is the image size (see patch size column in Table 1),
        and `T` is the number of time steps in the image sequence.

        >>> loader = SEVIRReaderHDF5(...)
        >>> nlwt = pd.IndexSlice[0:1, :, :, 0] # (N, L, W, T)
        >>> loader[("S728503", "vis"), nlwt]
        """
        # img_id, img_t, file_index = index
        # idx = self.loc[index]
        # print(idx)
        # return self._files[idx]

        return super().__getitem__(index)

    def groupby(
        self, by: list[typing.Literal["id", "img_type", "file_index"]]
    ) -> SeriesGroupBy[int, tuple[str | int, ...]]:
        return self._index.groupby(by=by)

    def select(
        self, key: typing.Iterable[tuple[str, str, int]], time: int | slice = slice(None)
    ) -> list[NDArray[np.floating[typing.Any]]]:
        return [self.pick(key, time=time) for key in key]

    def pick(self, key: tuple[str, str, int], *, time: int | slice = slice(None)) -> NDArray[np.floating[typing.Any]]:
        img_type, img_type, file_index = key
        idx = self.loc[key]
        h5 = self._store[idx]
        return h5
        # return self.loc[key]

        # if img_type == LIGHTNING:
        #     return h5[file_index][:]

        # return h5.data[file_index : file_index + 1, :, :, time]  # type: ignore

    def get_file(self, key: tuple[str, str, int]) -> H5File[pd.DataFrame]:
        return self._store[self.loc[key]]


def main() -> None:
    from .catalog import Catalog

    store = H5Store(Catalog())
    print(store["R18032123577290", "ir069", 166])

    # print(store.index)
    # print(store)
    # print(store.invert_index())


if __name__ == "__main__":
    main()
