from __future__ import annotations

import argparse
import multiprocessing.pool
import os
import typing

import h5py
import numpy as np
import pandas as pd
import tqdm
import zarr
from numpy.typing import NDArray

from .catalog import SEVIRCatalog
from .constants import (
    DEFAULT_FRAME_TIMES,
    FILE_NAME,
    ID,
    IMG_TYPE,
    LIGHTNING,
    TIME_UTC,
    SEVIRImageType,
)

ixs = pd.IndexSlice
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

    def __init__(self, path: str, types: list[SEVIRImageType]) -> None:
        self.store = store = zarr.DirectoryStore(path)
        self.root = root = zarr.group(store=store, overwrite=True)
        self.groups = {stype: root.create_group(stype) for stype in types}

    def update(self, stype: SEVIRImageType, img_id: str, data: NDArray, metadata: Metadata) -> SEVIRStoreRoot:
        self.groups[stype].array(img_id, data)
        self.groups[stype].attrs[img_id] = metadata
        return self

    def h5_loader(self, metadata: Metadata) -> None:
        stype, sid = (metadata[IMG_TYPE], metadata[ID])

        with h5py.File(metadata[FILE_NAME], "r") as f:
            arr = np.array(f[stype]) if stype != LIGHTNING else reshape_lightning_data(f[sid][:])  # type: ignore
        self.update(stype, sid, arr, metadata)

    def batch_update(self, records: list[Metadata], processes: int = 4) -> None:
        p_bar = tqdm.tqdm(total=len(records))
        with multiprocessing.pool.Pool(processes=processes) as pool:
            for _ in pool.imap_unordered(self.h5_loader, records):
                p_bar.update()


def main(types: list[SEVIRImageType], wrk_dir: str) -> None:
    cat = SEVIRCatalog(
        os.path.join(wrk_dir, "CATALOG.csv"),
        prefix=os.path.join(wrk_dir, "data"),
    )
    cat.validate_paths()
    df = cat.to_pandas().loc[ixs[:, types], :]
    df[TIME_UTC] = df[TIME_UTC].dt.strftime("%Y-%m-%d %H:%M:%S")
    root = SEVIRStoreRoot(os.path.join(wrk_dir, "zarr.array"), types)
    root.batch_update(df.reset_index().to_dict("records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", type=SEVIRImageType, nargs="+", default=[member for member in SEVIRImageType])
    parser.add_argument("wrk_dir", type=str, help="path to the directory containing the SEVIR dataset")
    main(**vars(parser.parse_args()))
