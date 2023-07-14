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
from .constants import FILE_NAME, ID, IMG_TYPE, LIGHTNING, ImageType, ImageTypeValue, FILE_INDEX
from ._typing import ImageIndexerType

from .lib import sel


class SEVIRFileHDF5(h5py.File):
    """subclass of h5py.File that provides a few convenience methods for SEVIR files and __reduce__ to allow
    for pickling which is required for multiprocessing."""

    __slots__ = ("key",)
    if typing.TYPE_CHECKING:
        image_id: str
        image_type: ImageType
        key: ImageTypeValue | str
        # data: h5py.Dataset
        __getitem__: typing.Callable[[int | str | slice], h5py.Dataset]

    def __init__(self, filename: str, key: str, mode: str = "r") -> None:
        super().__init__(filename, mode)
        self.key = key
        # self.data = typing.cast(h5py.Dataset, super().__getitem__(key))

    # def __reduce__(self) -> tuple[typing.Type[SEVIRFileHDF5], tuple[typing.Any, ...]]:
    #     return (SEVIRFileHDF5, (self.filename, self.image_id, self.image_type))

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data(self) -> h5py.Dataset:
        return self[self.key]  # type: ignore

    # def __getitem__(self, index: tuple[ImageIndexerType, ...]) -> NDArray[np.floating[typing.Any]]:
    #     k, idx = index
    #     return super().__getitem__(k)[idx]


_FILE_STORE_INDEX = "<FILE_STORE>"


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


def _h5_executor(idx, df):
    file, key = idx
    if key == LIGHTNING:
        key, *_ = df[ID]
    return SEVIRFileHDF5(file, key, mode="r"), df


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
        event_id, img_type, file_index = key
        h5 = self._store[self.loc[key]]
        if event_id == LIGHTNING:
            return h5[file_index][:]

        return h5[img_type][file_index : file_index + 1, :, :, time]  # type: ignore


def main() -> None:
    from .catalog import Catalog

    reader = FileReader(Catalog())

    print(reader.index)
    print(reader.shapes)
    print(reader.invert_index())


if __name__ == "__main__":
    main()
