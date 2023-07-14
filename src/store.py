"""
A script to convert the SEVIR h5 dataset into a zarr dataset.

```bash
python -m src.dataset --types=ir069 /mnt/nuc/c/sevir
```
"""
from __future__ import annotations

import argparse
import multiprocessing.pool
import os
import sys
import typing

import h5py
import numpy as np
import pandas as pd
import tqdm
import zarr
from numpy.typing import NDArray

from .catalog import Catalog
from .constants import (
    DEFAULT_FRAME_TIMES,
    FILE_NAME,
    ID,
    IMG_TYPE,
    LIGHTNING,
    TIME_UTC,
    ImageType,
)

Metadata: typing.TypeAlias = dict[typing.Hashable, typing.Any]


def reshape_lightning_data(data: np.ndarray, t_slice=slice(0, None)) -> NDArray[np.int16]:
    """Converts Nx5 lightning data matrix into a 2D grid of pixel counts."""

    out_size = (48, 48, len(DEFAULT_FRAME_TIMES)) if t_slice.stop is None else (48, 48, 1)
    if data.shape[0] == 0:
        return np.zeros((1,) + out_size, dtype=np.int16)

    # filter out points outside the grid
    x, y = data[:, 3], data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[m, :]
    if data.shape[0] == 0:
        return np.zeros((1,) + out_size, dtype=np.int16)

    # Filter/separate times
    t = data[:, 0]
    if t_slice.stop is not None:  # select only one time bin
        if t_slice.stop > 0:
            if t_slice.stop < len(DEFAULT_FRAME_TIMES):
                tm = np.logical_and(t >= DEFAULT_FRAME_TIMES[t_slice.stop - 1], t < DEFAULT_FRAME_TIMES[t_slice.stop])
            else:
                tm = t >= DEFAULT_FRAME_TIMES[-1]
        else:  # special case:  frame 0 uses lght from frame 1
            tm = np.logical_and(t >= DEFAULT_FRAME_TIMES[0], t < DEFAULT_FRAME_TIMES[1])

        data = data[tm, :]
        z = np.zeros(data.shape[0], dtype=np.int64)
    else:  # compute z coordinate based on bin location times
        z = np.digitize(t, DEFAULT_FRAME_TIMES) - 1
        z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index([y, x, z], out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]


class SEVIRStoreRoot:
    __slots__ = ("store", "root", "groups")

    def __init__(self, path: str, types: list[ImageType]) -> None:
        self.store = store = zarr.DirectoryStore(path)
        self.root = root = zarr.group(store=store, overwrite=True)
        self.groups = {img_t: root.create_group(img_t) for img_t in types}

    def update(self, img_t: ImageType, img_id: str, data: NDArray, metadata: Metadata) -> SEVIRStoreRoot:
        self.groups[img_t].array(img_id, data)
        self.groups[img_t].attrs[img_id] = metadata
        return self

    def h5_loader(self, metadata: Metadata) -> None:
        img_type, img_id = metadata[IMG_TYPE], metadata[ID]

        with h5py.File(metadata[FILE_NAME], "r") as f:
            arr = np.array(f[img_type]) if img_type != LIGHTNING else reshape_lightning_data(f[img_id][:])  # type: ignore[unused-ignore]
        self.update(img_type, img_id, arr, metadata)

    def batch_update(self, records: list[Metadata], processes: int | None = None) -> None:
        p_bar = tqdm.tqdm(total=len(records))
        with multiprocessing.pool.Pool(processes=processes) as pool:
            for _ in pool.imap_unordered(self.h5_loader, records):
                p_bar.update()


def main(*, img_types: list[ImageType], wrk_dir: str, nproc: int) -> None:
    df = (
        Catalog(
            os.path.join(wrk_dir, "CATALOG.csv"),
            data_dir="data",
        )
        .validate()
        .to_pandas()
        .loc[pd.IndexSlice[:, img_types], :]
    )

    df[TIME_UTC] = df[TIME_UTC].dt.strftime("%Y-%m-%d %H:%M:%S")
    root = SEVIRStoreRoot(os.path.join(wrk_dir, "zarr.array"), img_types)
    root.batch_update(df.reset_index().to_dict("records"), processes=nproc)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-types",
        type=ImageType,
        nargs="+",
        default=[member for member in ImageType],
        help="SEVIR image types to include",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="number of processes to use for parallel loading",
    )
    parser.add_argument(
        "wrk_dir",
        type=str,
        help="path to the directory containing the SEVIR dataset",
    )
    sys.exit(main(**vars(parser.parse_args())))
