from __future__ import annotations

import contextlib
import logging
import os
import random
import typing
from typing import Final, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler

from .catalog import Catalog
from .constants import IR_069, IR_107, LGHT, VIL, VIS, ImageType
from .h5 import H5Store

logging.getLogger().setLevel(logging.INFO)

idx = pd.IndexSlice


class SEVIRGenerator(IterableDataset[tuple[Tensor, Tensor]]):
    __slots__ = ("store", "image_ids", "x_img_types", "y_img_types", "x", "y")
    if typing.TYPE_CHECKING:
        store: Final[H5Store]
        image_ids: Final[list[str]]
        x_img_types: Final[list[ImageType]]
        y_img_types: Final[list[ImageType]]
        x: Final[Catalog]
        y: Final[Catalog]

    def __init__(
        self,
        sevir: Catalog | str,
        *,
        inputs: Iterable[ImageType],
        features: Iterable[ImageType],
        catalog: str | None = None,
        data_dir: str | None = None,
        nproc: int | None = os.cpu_count(),
    ) -> None:
        super().__init__()
        image_set = set(inputs).union(features)
        self.meta = meta = (
            sevir
            if isinstance(sevir, Catalog)
            else Catalog(sevir, img_types=image_set, catalog=catalog, data_dir=data_dir)
        )

        self.x, self.y = x, y = meta.split_by_types(list(inputs), list(features))
        self.image_ids = meta.id.unique().to_list()
        self.x_img_types, self.y_img_types = list(x.image_set), list(y.image_set)
        assert meta.id.n_unique() == x.id.n_unique() == y.id.n_unique()
        assert len(x.image_set) + len(y.image_set) == len(image_set)
        self.store = H5Store(meta, nproc=nproc)

    @typing.overload
    def get_batch(
        self,
        img_id: int | str | bytes | None = None,
        img_type: list[ImageType] | None = None,
        metadata: typing.Literal[False] = ...,
    ) -> tuple[Tensor, Tensor]:
        ...

    @typing.overload
    def get_batch(
        self,
        img_id: int | str | bytes | None = None,
        img_type: list[ImageType] | None = None,
        metadata: typing.Literal[True] = ...,
    ) -> tuple[tuple[Tensor, Tensor], pl.DataFrame]:
        ...

    def get_batch(
        self,
        img_id: int | str | bytes | None = None,
        img_type: list[ImageType] | None = None,
        metadata: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], pl.DataFrame]:
        if img_id is None:
            img_id = random.choice(self.image_ids)
        if isinstance(img_id, int):
            img_id = self.image_ids[img_id]

        x = np.array(self.store[img_id, img_type or self.x_img_types])
        y = np.array(self.store[img_id, self.y_img_types])
        values = torch.from_numpy(x), torch.from_numpy(y)
        if not metadata:
            return values
        return (values, self.meta.data.filter(self.meta.id == img_id))

    @typing.overload
    def iter_batches(
        self,
        *,
        metadata: typing.Literal[False] = ...,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        ...

    @typing.overload
    def iter_batches(
        self,
        *,
        metadata: typing.Literal[True] = ...,
    ) -> Iterator[tuple[tuple[Tensor, Tensor], pl.DataFrame]]:
        ...

    def iter_batches(
        self,
        *,
        metadata: bool = False,
    ) -> Iterator[tuple[Tensor, Tensor]] | Iterator[tuple[tuple[Tensor, Tensor], pl.DataFrame]]:
        bar = tqdm.tqdm(total=len(self.image_ids))
        for img_id in self.image_ids:
            yield self.get_batch(img_id, metadata=metadata)  # type: ignore
            bar.update(1)
        bar.close()

    def close(self) -> None:
        logging.info("Closing SEVIRstoreHDF5")
        self.store.close_all()

    @contextlib.contextmanager
    def session(self):
        try:
            yield self
        finally:
            self.close()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_batch(index)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        try:
            for img_id in self.image_ids:
                yield self.get_batch(img_id)

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            self.close()
            raise StopIteration


class SEVIRLoader(DataLoader[tuple[Tensor, Tensor]]):
    if typing.TYPE_CHECKING:
        dataset: SEVIRGenerator

    def __init__(
        self,
        dataset: SEVIRGenerator,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: typing.Callable[[list[tuple[Tensor, Tensor]]], typing.Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: typing.Callable[[int], None] | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def __enter__(self) -> SEVIRLoader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.dataset.close()


def main(
    sevir="/mnt/nuc/c/sevir",
    data_dir: str = "data",
    catalog: str = "CATALOG.csv",
    inputs={IR_069, IR_107},
    features={VIL},
) -> None:
    logging.info(
        f"""\
⛈️  SEVIRLoader Example ⛈️
{VIL=}
{IR_069=}
{IR_107=}
{VIS=}
{LGHT=}
 
"""
    )

    with SEVIRGenerator(
        sevir, catalog=catalog, data_dir=data_dir, inputs=inputs, features=features
    ).session() as generator:
        i = 0
        for (l, r), df in generator.iter_batches(metadata=True):
            print(df, l.shape, r.shape, sep="\n")
            if i > 10:
                break
            i += 1


if __name__ == "__main__":
    main()
