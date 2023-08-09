from __future__ import annotations

import logging
import multiprocessing.pool
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Iterator,
    Literal,
    Mapping,
    cast,
    overload,
)

import h5py
import numpy as np
import pandas as pd
import polars as pl
import tqdm
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from .._typing import AnyT, Array, N, Nd
from ..constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_FRAME_TIMES,
    DEFAULT_PATCH_SIZE,
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
from ..generic import AbstractCatalog, AbstractContextManager
from .catalog import Catalog


# =====================================================================================================================
def squarespace(
    in_size: int, out_size: int
) -> tuple[tuple[Array[Nd[N], Any], Array[Nd[N], Any]], tuple[Array[Nd[N, N], Any], Array[Nd[N, N], Any]]]:
    """
    >>> import numpy as np
    >>> import sevir.core.h5
    >>> points, grid = sevir.core.h5.squarespace(4,6)
    >>> points
    (array([0.        , 0.08333333, 0.16666667, 0.25      ]), array([0.        , 0.08333333, 0.16666667, 0.25      ]))
    >>> grid
    (array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
           [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
           [0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 ],
           [0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
           [0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 ],
           [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]), array([[0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25],
           [0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25]]))
    """
    xy1 = np.linspace(0, 1.0 / in_size, in_size)
    xy2 = np.linspace(0, 1.0 / in_size, out_size)
    return (xy1, xy1), tuple(np.meshgrid(xy2, xy2, indexing="ij"))  # type: ignore[return-value]


def interpatch(arr: Array[Nd[N, N, N], AnyT], *, patch_size: int) -> Array[Nd[N, N, N], AnyT]:
    """
    Interpolate the values of a 4D array to a new PatchSize

    assumes a 4D array with shape (B, L, W, T) where L == W, reshapes the array
    and interpolates the array to (B, lxl, lxl, T) using
    scipy.interpolate.RegularGridInterpolator.

    >>> import numpy as np
    >>> import sevir.core.h5
    >>> arr = np.random.randint(0, 255, (384, 384, 49))
    >>> sevir.core.h5.interpatch(arr, 768).shape
    (768, 768, 49)
    """
    x, y = arr.shape[:2]
    if not x == y:
        raise ValueError(f"array must be square, but got shape: {arr.shape}")
    points, values = squarespace(x, patch_size)
    interp = RegularGridInterpolator(points, arr)
    return interp(values).astype(arr.dtype)


def reshape_lightning(
    data: Array[Nd[N, Literal[5]], Any],
    *,
    patch_size: int = 48,
    t_slice=slice(0, None),
    t_frame: Array[Nd[N], np.float64] = DEFAULT_FRAME_TIMES,
) -> Array[Nd[N, N, N], np.int16]:
    """Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        this function was adopted from
        [eie-sevir](https://github.com/MIT-AI-Accelerator/eie-sevir/blob/master/sevir/generator.py#L386)

    >>> import numpy as np
    >>> import sevir.core.h5
    >>> a = np.random.rand(3, 5)
    >>> sevir.core.h5.reshape_lightning(a, patch_size=256).shape
    (256, 256, 49)
    >>> sevir.core.h5.reshape_lightning(a, patch_size=256, t_slice=slice(0,1)).shape
    (256, 256, 1)
    """
    shape = nx, ny, _ = (patch_size, patch_size, len(t_frame) if t_slice.stop is None else 1)

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
    return np.bincount(lwt, minlength=np.prod(shape)).reshape(shape).astype(np.int16)


class H5File(h5py.File):
    """subclass of h5py.File that provides a few convenience methods for SEVIR files and __reduce__ to allow
    for pickling which is required for multiprocessing."""

    __slots__ = ("image_type",)
    if TYPE_CHECKING:
        image_type: ImageType

    def __init__(self, filename: str, image_type: ImageType) -> None:
        self.image_type = image_type
        super().__init__(filename, mode="r")

    def __getitem__(self, __id: bytes | str, /) -> h5py.Dataset:
        img_t = self.image_type

        return cast(h5py.Dataset, super().__getitem__(__id if img_t == LGHT else img_t))

    def select(self, id_: str, fidx: int) -> Array[Nd[N, N, N], np.int16]:
        ds = self[id_]
        return reshape_lightning(ds[...]) if self.image_type == LGHT else ds[fidx, ...]

    @property
    def event_ids(self) -> Array[Nd[N], np.bytes_]:
        return super().__getitem__(ID)[...]  # type: ignore[unused-ignore]


class Store(
    Mapping[str | bytes, list[Array[Nd[N, N, N], np.int16]]], AbstractCatalog, AbstractContextManager["Store"]
):
    """
    the dataset effectively maps the following structure:
    >>> data[FILE_REF][IMG_TYPE][EVENT_ID][FILE_INDEX, L, W, T]
    this class aggregates the data from multiple files that can be grouped
    by the same event_id
    """

    __slots__ = ("_files", "catalog", "normalization")
    if TYPE_CHECKING:
        _files: list[H5File]
        catalog: Catalog
        normalization: dict[ImageType, tuple[int, int]] | None

    # - Initialization
    def __init__(
        self,
        catalog: Catalog | pl.DataFrame | pd.DataFrame,
        normalization: dict[ImageType, tuple[int, int]] | None = None,
    ) -> None:
        if isinstance(catalog, (pl.DataFrame, pd.DataFrame)):
            catalog = Catalog(catalog)

        # create a copy of the catalog and null out the reference column
        self.catalog = cat = catalog.with_reference(None, inplace=False)
        self.normalization = normalization

        if not set(cat.columns).issuperset([FILE_NAME, ID, IMG_TYPE, FILE_INDEX]):
            raise ValueError(f"Catalog is missing columns: {[FILE_NAME, ID, IMG_TYPE, FILE_INDEX]}")

        logging.info(f"Loading {cat.file_name.n_unique()} files with {cat.img_type.n_unique()} image types.")
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
    ) -> Store:
        return cls(Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir))

    # - AbstractCatalog interface
    @property
    def data(self) -> pl.DataFrame:
        return self.catalog.data

    @property
    def types(self) -> tuple[ImageType, ...]:
        return self.catalog.types

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()
        self.catalog.close()

    def is_closed(self) -> bool:
        return len(self._files) == 0

    # =================================================================================================================
    def pick(self, img_type: ImageType, fref: int, id_: str, fidx: int) -> Array[Nd[N, N, N], np.int16]:
        arr = self._files[fref].select(id_, fidx)
        if self.normalization is not None:
            arr = self.normalize(arr, *self.normalization[img_type])
        return arr

    # - Methods
    @overload  # type: ignore[misc]
    def select(
        self, id_: str | bytes, *, img_types: Collection[ImageType] | None = None, metadata: Literal[False] = False
    ) -> list[Array[Nd[N, N, N], np.int16]]:
        ...

    @overload
    def select(
        self, id_: str | bytes, *, img_types: Collection[ImageType] | None = None, metadata: Literal[True] = True
    ) -> tuple[list[Array[Nd[N, N, N], np.int16]], pl.DataFrame]:
        ...

    def select(
        self, id_: str | bytes, *, img_types: Collection[ImageType] | None = None, metadata: bool = False
    ) -> list[Array[Nd[N, N, N], np.int16]] | tuple[list[Array[Nd[N, N, N], np.int16]], pl.DataFrame]:
        if isinstance(id_, bytes):
            id_ = id_.decode("utf-8")
        arrays = [
            self.pick(img_t, fref, id_, fidx) for img_t, fref, fidx in self.iter_indices(id_, img_types=img_types)
        ]

        if metadata:
            return arrays, self.data.filter(self.id == id_)
        return arrays

    # - Mapping interface
    def __getitem__(self, id_: str | bytes) -> list[Array[Nd[N, N, N], np.int16]]:
        return self.select(id_)

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[bytes]:
        return iter(set(self.catalog.id))

    def iter_files(self) -> Iterator[tuple[str, ImageType]]:
        columns = pl.col(FILE_NAME), pl.col(IMG_TYPE)
        df = self.data.select(columns)
        return df.unique(subset=FILE_NAME).iter_rows()  # type: ignore[return-value]

    def iter_indices(
        self, id_: str, *, img_types: Collection[ImageType] | None = None
    ) -> Iterator[tuple[ImageType, int, int]]:
        """returns an iterator of (file_ref, file_index) tuples for the given id and image types"""
        img_types = img_types or self.types
        columns = pl.col(IMG_TYPE), pl.col(FILE_REF), pl.col(FILE_INDEX)
        df = self.data.select(columns)
        return (
            df.filter((self.catalog.id == id_) & (self.img_type.is_in(img_types))).sort(
                pl.col(IMG_TYPE).map_dict({t: i for i, t in enumerate(img_types)}), descending=False
            )
            # .drop(IMG_TYPE)
            .iter_rows()  # type: ignore[return-value]
        )

    # =================================================================================================================
    def interp_stack(
        self, img_id: str | bytes, *, patch_size: int = DEFAULT_PATCH_SIZE
    ) -> Array[Nd[N, N, N, N], np.int16]:
        """Interpolates `(X, Y)` to the desired patch size
        `list[Array[X, Y, T]] -> Array[C, X, Y, T]` to the provided patch_size"""
        arrays = [
            arr if img_t.patch_size == patch_size else interpatch(arr, patch_size=patch_size)
            for img_t, arr in zip(self.types, self.select(img_id))
        ]
        return np.stack(arrays, axis=0)

    def to_xarray(
        self,
        img_ids: str | list[str],
        *,
        patch_size: int = DEFAULT_PATCH_SIZE,
    ) -> xr.Dataset:
        if not isinstance(img_ids, list):
            img_ids = [img_ids]

        data_vars = {}

        # logging.info(f"🌩️ Interpolating {len(img_ids)} images to patch size: {patch_size} 🌩️")
        # bar = tqdm.tqdm(total=len(img_ids))
        for id_ in img_ids:
            data_vars[id_] = (["c", "x", "y", "t"], self.interp_stack(id_, patch_size=patch_size))
        #   bar.update(1)
        # bar.close()

        coords = {
            "channel": (["c"], [str(t) for t in self.types]),
            "patch": (["x", "y"], np.arange(0, patch_size * patch_size).reshape(patch_size, patch_size)),
            "time": (["t"], DEFAULT_FRAME_TIMES),
        }

        attrs = {
            "description": "SEVIR Dataset with all channels interpolated to the same patch size.",
            "patch_size": patch_size,
            # "img_ids": img_ids,
        } | {str(t): {"description": t.description, "sensor": t.sensor} for t in self.types}

        return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    def to_zarr(
        self,
        dest: str,
        img_ids: list[str] | Array[Nd[N], np.str_] | pl.Series,
        *,
        patch_size: int = DEFAULT_PATCH_SIZE,
        n_chunk: int = 1,
        n_proc: int | None = None,
    ) -> None:
        img_ids = np.unique(img_ids)
        assert len(img_ids.shape) == 1
        # total = img_ids.size // (chunks)
        n_chunk = max(len(img_ids) // n_chunk, 1)
        arrays = np.array_split(img_ids, n_chunk)
        logging.info(
            f"🌩️ Interpolating {len(img_ids)} images to patch size: {patch_size} 🌩️"
            f" and writing to {dest} with {n_chunk} chunks."
        )
        with multiprocessing.pool.ThreadPool(n_proc) as pool:
            for ds in tqdm.tqdm(
                pool.imap_unordered(lambda arr: self.to_xarray(arr.tolist(), patch_size=patch_size), arrays),
                total=len(arrays),
            ):
                mode = "w" if not os.path.exists(dest) else "a"

                ds.to_zarr(dest, mode=mode)  # type:ignore

    @staticmethod
    def normalize(arr: np.ndarray, scale: int, offset: int) -> np.ndarray:
        """
        Normalized data using s = (scale,offset) via Z = (X-offset)*scale
        """
        return (arr - offset) * scale

    @staticmethod
    def unnormalize(arr: np.ndarray, scale: int, offset: int) -> np.ndarray:
        """
        Reverses the normalization performed in a SEVIRGenerator generator
        given s=(scale,offset)
        """
        return arr / scale + offset
