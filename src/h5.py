from __future__ import annotations

import logging
import multiprocessing.pool
import os
import typing

import h5py
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
import tqdm
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .catalog import Catalog
from .constants import (
    FILE_NAME,
    ID,
    IMG_TYPE,
    LIGHTNING,
    ImageType,
    ImageTypeValue,
    FILE_INDEX,
    DEFAULT_FRAME_TIMES as FRAME_TIMES,
)
from ._typing import ImageIndexerType

from .lib import sel


def _lght_to_grid(data: NDArray, t_slice=slice(0, None)):
    """
    Converts Nx5 lightning data matrix into a 2D grid of pixel counts
    """
    # out_size = (48,48,len(FRAME_TIMES)-1) if isinstance(t_slice,(slice,)) else (48,48)
    out_size = (48, 48, len(FRAME_TIMES)) if t_slice.stop is None else (48, 48, 1)
    if data.shape[0] == 0:
        return np.zeros((1,) + out_size, dtype=np.float32)

    # filter out points outside the grid
    x, y = data[:, 3], data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[m, :]
    if data.shape[0] == 0:
        return np.zeros((1,) + out_size, dtype=np.float32)

    # Filter/separate times
    t = data[:, 0]
    if t_slice.stop is not None:  # select only one time bin
        if t_slice.stop > 0:
            if t_slice.stop < len(FRAME_TIMES):
                tm = np.logical_and(t >= FRAME_TIMES[t_slice.stop - 1], t < FRAME_TIMES[t_slice.stop])
            else:
                tm = t >= FRAME_TIMES[-1]
        else:  # special case:  frame 0 uses lght from frame 1
            tm = np.logical_and(t >= FRAME_TIMES[0], t < FRAME_TIMES[1])
        # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

        data = data[tm, :]
        z = np.zeros(data.shape[0], dtype=np.int64)
    else:  # compute z coodinate based on bin locaiton times
        z = np.digitize(t, FRAME_TIMES) - 1
        z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]


LENGTH = WIDTH = 48
LIGHTNING_TIME = 0
"""0	Time of flash in seconds relative to time_utc column in the catalog."""
X_DEG = 1
"""1	Reported latitude of flash in degrees"""
Y_DEG = 2
"""2	Reported longitude of flash in degrees"""
RASTER_X = 3
"""3	Flash X coordinate when converting to raster"""
RASTER_Y = 4
"""4	Flash Y coordinate when converting to raster"""


def _lght_to_grid_new(data: NDArray, t_slice=slice(0, None)) -> NDArray:
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
    print(data.shape)
    shape = length, width, time = (LENGTH, WIDTH, len(FRAME_TIMES) if t_slice.stop is None else 1)

    if data.shape[0] == 0:  # there are no samples
        return np.zeros((1,) + shape, dtype=np.float32)

    # filter out points outside the grid
    x, y = data[:, RASTER_X], data[:, RASTER_Y]
    mask = np.logical_and.reduce([x >= 0, x < length, y >= 0, y < width])
    data = data[mask, :]

    t = data[:, LIGHTNING_TIME]
    # Filter/separate times
    if t_slice.stop is not None:  # select only one time bin
        if t_slice.stop > 0:
            if t_slice.stop < len(FRAME_TIMES):
                time_mask = np.logical_and(t >= FRAME_TIMES[t_slice.stop - 1], t < FRAME_TIMES[t_slice.stop])
            else:
                time_mask = t >= FRAME_TIMES[-1]
        else:  # special case:  frame 0 uses lght from frame 1
            time_mask = np.logical_and(t >= FRAME_TIMES[0], t < FRAME_TIMES[1])

        data = data[time_mask, :]
        z = np.zeros(data.shape[0], dtype=np.int64)
    else:  # compute z coordinate based on bin location times
        z = np.digitize(t, FRAME_TIMES) - 1
        z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), shape)
    n = np.bincount(k, minlength=np.prod(shape))
    return np.reshape(n, shape).astype(np.int16)[np.newaxis, :]


class SEVIRFileHDF5(h5py.File):
    """subclass of h5py.File that provides a few convenience methods for SEVIR files and __reduce__ to allow
    for pickling which is required for multiprocessing."""

    __slots__ = ("_img_type", "_metadata")
    if typing.TYPE_CHECKING:
        _img_type: ImageType
        _metadata: pd.DataFrame | None

    def __init__(self, filename: str, img_type: str | None = None, metadata: pd.DataFrame | None = None) -> None:
        super().__init__(filename, mode="r")
        self._img_type = ImageType(img_type)
        self._metadata = metadata

    @property
    def data(self) -> typing.Mapping[tuple[typing.Any, ...], NDArray]:
        return (
            self[self._img_type]
            if self._img_type != LIGHTNING
            else [
                [_lght_to_grid(self[id_], sel[5:10]), _lght_to_grid_new(self[id_], sel[5:10])] for id_ in self[ID][:1]
            ]  # [...]
        )  # self[self[ID][...][0]][:, :]

    @property
    def ids(self) -> NDArray[np.bytes_]:
        return self[ID][...]


class FileDatastore(Dataset[NDArray[np.floating[typing.Any]]]):
    """The h5py.File object does not like being inserted into a pandas DataFrame or a numpy array. This class
    provides a wrapper around h5py.File that allows for this. It also provides a convenient way to index into
    the file using a pandas DataFrame."""

    __slots__ = ("_index", "_store", "loc")
    if typing.TYPE_CHECKING:
        _index: pd.Series[int]
        _store: list[SEVIRFileHDF5]

    def __init__(self, index: pd.MultiIndex) -> None:
        self._index = pd.Series(index=index, name=self.index_name, dtype="Int64")
        self._store = []
        self.loc = self._index.loc

    def __setitem__(self, key: np.ndarray, value: SEVIRFileHDF5) -> None:
        self._index[key] = len(self._store)
        self._store.append(value)

    @property
    def index_name(self) -> str:
        return f"<{self.__class__.__name__}.index>"

    @property
    def index(self) -> pd.MultiIndex:
        return self._index.index  # type: ignore

    def invert_index(self) -> pd.DataFrame:
        return self._index.reset_index().set_index(self.index_name)

    @property
    def n_event(self) -> pd.Series[int]:
        # TODO: there is something weird with the dataset and the N dim is not always the same
        # which is not really expected.
        return self.shapes.groupby(level=0)["N"].min()

    @property
    def shapes(self) -> pd.DataFrame:
        return pd.DataFrame(
            [file.data.shape for file in self._store],
            columns=["N", "L", "W", "T"],
        )

    def groupby(
        self, by: list[typing.Literal["id", "img_type", "file_index"]]
    ) -> SeriesGroupBy[int, tuple[str | int, ...]]:
        return self._index.groupby(by=by)

    def close_all(self) -> None:
        for f in self._store:
            f.close()
        self._store.clear()
        self._index = self._index.iloc[0:0]


def _h5_executor(idx: tuple[str, str], df: pd.DataFrame):
    file_name, key = idx
    if key == LIGHTNING:
        key, *_ = df[ID]
    return SEVIRFileHDF5(file_name, key, mode="r"), df


class FileReader(FileDatastore):
    __slots__ = ("data",)

    def __init__(self, catalog: Catalog, nproc: int | None = os.cpu_count()) -> None:
        # validate the catalog before attempting to open files
        df = catalog.validate().to_pandas().reset_index(drop=False)
        super().__init__(pd.MultiIndex.from_frame(df[[ID, IMG_TYPE, FILE_INDEX]]))

        group_iterator = typing.cast(
            typing.Iterable[tuple[tuple[str, str], pd.DataFrame]],
            df[[FILE_NAME, ID, IMG_TYPE, FILE_INDEX]].groupby(by=[FILE_NAME, IMG_TYPE]),
        )
        logging.info("Opening HDF5 files")
        with multiprocessing.pool.ThreadPool(nproc) as pool:
            bar = tqdm.tqdm(total=df.file_name.nunique())
            for file, event in pool.starmap(_h5_executor, group_iterator):
                self[event[[ID, IMG_TYPE, FILE_INDEX]].to_numpy()] = file
                bar.update(1)

    def __getitem__(
        self, index: tuple[tuple[str, str], tuple[ImageIndexerType, ...]]
    ) -> NDArray[np.floating[typing.Any]]:
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
        return self.select(sel[index])
        # key, nlwt = index  # (id, img_type), (n, l, w, t)
        # return self.data[key][nlwt]

    # def __len__(self) -> int:
    #     return sum(len(file) for file in self.data.values())

    # @property
    # def index(self) -> pd.MultiIndex:
    #     return self._store.index

    # # @property
    # # def size(self) -> dict[tuple[str, str | ImageType], int]:
    # #     return {key: len(file) for key, file in self.data.items()}

    # @property
    # def n_event(self) -> pd.Series[int]:
    #     # TODO: there is something weird with the dataset and the N dim is not always the same
    #     # which is not really expected.
    #     return self.shapes.groupby(level=0)["N"].min()

    # @property
    # def shapes(self) -> pd.DataFrame:
    #     return self._store.shapes

    # def keys(self) -> list[tuple[str, str | ImageType]]:
    #     return list(self.data.keys())

    # def get_event_count(self, img_id: str) -> int:
    #     for img_id, img_t in self.data.keys():
    #         ds = self.data[(img_id, img_t)].data
    #         if img_id == img_id:
    #             x, *_ = ds.shape
    #             return x
    #     else:
    #         raise KeyError(f"Event {img_id} not found in dataset")

    # def close_all(self) -> None:
    #     self.store.close_all()

    def select(
        self, key: typing.Iterable[tuple[str, str, int]], time: int | slice = slice(None)
    ) -> list[NDArray[np.floating[typing.Any]]]:
        return [self.pick(key, time=time) for key in key]

    def pick(self, key: tuple[str, str, int], *, time: int | slice = slice(None)) -> NDArray[np.floating[typing.Any]]:
        img_type, img_type, file_index = key
        h5 = self._store[self.loc[key]]
        if img_type == LIGHTNING:
            return h5[file_index][:]

        return h5.data[file_index : file_index + 1, :, :, time]  # type: ignore


def main() -> None:
    from .catalog import Catalog

    reader = FileReader(Catalog())

    print(reader.index)
    print(reader.shapes)
    print(reader.invert_index())


if __name__ == "__main__":
    main()
