from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Collection,
    Final,
    Iterator,
    Literal,
    Mapping,
    cast,
    overload,
)

import h5py
import numpy as np
import polars as pl
import tqdm
from numpy.typing import NDArray

from ._typing import AnyT, Array, N, Nd
from .catalog import AbstractCatalog, Catalog
from .constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_FRAME_TIMES,
    DEFAULT_PATH_TO_SEVIR,
    FILE_INDEX,
    FILE_NAME,
    FILE_REF,
    FLASH_TIME,
    FLASH_X,
    FLASH_Y,
    ID,
    IMG_TYPE,
    LGHT,
    ImageType,
)
from .generic import AbstractContextManager


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
        self.image_type = image_type
        super().__init__(filename, mode="r")

    def __getitem__(self, __id: bytes | str, /) -> Array[Nd[N, N, N, N], np.int16]:
        img_t = self.image_type

        return cast(
            Array[Nd[N, N, N, N], np.int16],
            reshape_lightning_data(super().__getitem__(__id)) if img_t == LGHT else super().__getitem__(img_t),  # type: ignore[unused-ignore] # noqa: E501
        )

    def get_by_file_index(self, index: int) -> Array[Nd[N, N, N, N], np.int16]:
        return self[self.event_ids[index]]

    @property
    def event_ids(self) -> NDArray[np.bytes_]:
        return super().__getitem__(ID)[...]  # type: ignore[unused-ignore]


class H5Store(
    Mapping[str | bytes, list[Array[Nd[N, N, N, N], np.int16]]], AbstractCatalog, AbstractContextManager["H5Store"]
):
    """
    the dataset effectively maps the following structure:
    >>> data[FILE_REF][IMG_TYPE][EVENT_ID][FILE_INDEX, L, W, T]
    this class aggregates the data from multiple files that can be grouped
    by the same event_id
    """

    __slots__ = ("catalog", "_files")
    if TYPE_CHECKING:
        catalog: Catalog
        _files: list[H5File]

    # - Initialization
    def __init__(self, catalog: Catalog) -> None:
        # create a copy of the catalog and null out the reference column
        self.catalog = cat = catalog.with_reference(None, inplace=False)

        if not set(cat.columns).issuperset([FILE_NAME, ID, IMG_TYPE, FILE_INDEX]):
            raise ValueError(f"Catalog is missing columns: {[FILE_NAME, ID, IMG_TYPE, FILE_INDEX]}")

        logging.info(f"Loading {cat.file_name.n_unique()} files with {cat.image_types.n_unique()} image types.")
        self._files = []

        bar = tqdm.tqdm(total=cat.file_name.n_unique())

        for file_n, img_t in self.iter_files():
            cat.with_reference(
                pl.when(cat.file_name == file_n).then(len(self._files)).otherwise(cat.file_ref).alias(FILE_REF),
                inplace=True,
            )
            self._files.append(H5File(file_n, img_t))

            bar.update(1)
        bar.close()

    @classmethod
    def from_disk(
        cls,
        data: str = DEFAULT_PATH_TO_SEVIR,
        *,
        img_types: tuple[ImageType, ...] | None = None,
        catalog: str | None = DEFAULT_CATALOG,
        data_dir: str | None = DEFAULT_DATA,
    ) -> H5Store:
        return cls(Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir))

    # - AbstractCatalog interface
    @property
    def data(self) -> pl.DataFrame:
        return self.catalog.data

    @property
    def types(self) -> tuple[ImageType, ...]:
        return self.catalog.types

    # - Mapping interface
    def __getitem__(self, id_: str | bytes) -> list[Array[Nd[N, N, N, N], np.int16]]:
        return self.get_batch(id_)

    def __iter__(self) -> Iterator[bytes]:
        return iter(set(self.catalog.id))

    def __len__(self) -> int:
        return len(self._files)

    # - Methods
    @overload  # type: ignore[misc]
    def get_batch(
        self, id_: str | bytes, *, img_types: Collection[ImageType] | None = None, metadata: Literal[False] = False
    ) -> list[Array[Nd[N, N, N, N], np.int16]]:
        ...

    @overload
    def get_batch(
        self, id_: str | bytes, *, img_types: Collection[ImageType] | None = None, metadata: Literal[True] = True
    ) -> tuple[list[Array[Nd[N, N, N, N], np.int16]], pl.DataFrame]:
        ...

    def get_batch(
        self, id_: str | bytes, *, img_types: Collection[ImageType] | None = None, metadata: bool = False
    ) -> list[Array[Nd[N, N, N, N], np.int16]] | tuple[list[Array[Nd[N, N, N, N], np.int16]], pl.DataFrame]:
        if isinstance(id_, bytes):
            id_ = id_.decode("utf-8")

        batch = [
            self._files[fref][id_][fidx : fidx + 1, :, :, :]
            for fref, fidx in self.iter_indices(id_, img_types=img_types)
        ]
        if metadata:
            return batch, self.data.filter(self.id == id_)

        return batch

    def iter_files(self) -> Iterator[tuple[str, ImageType]]:
        columns = pl.col(FILE_NAME), pl.col(IMG_TYPE)
        df = self.data.select(columns)
        return df.unique(subset=FILE_NAME).iter_rows()

    def iter_indices(self, id_: str, *, img_types: Collection[ImageType] | None = None) -> Iterator[tuple[int, int]]:
        """returns an iterator of (file_ref, file_index) tuples for the given id and image types"""
        img_types = img_types or self.types
        columns = pl.col(IMG_TYPE), pl.col(FILE_REF), pl.col(FILE_INDEX)
        df = self.data.select(columns)
        return (
            df.filter((self.catalog.id == id_) & (self.image_types.is_in(img_types)))
            .sort(pl.col(IMG_TYPE).map_dict({t: i for i, t in enumerate(img_types)}), descending=False)
            .drop(IMG_TYPE)
            .iter_rows()  # type: ignore[return-value]
        )

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()
        self.catalog.close()

    def is_closed(self) -> bool:
        return len(self._files) == 0
