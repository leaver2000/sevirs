from __future__ import annotations

import logging
import os
import random
import typing
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler

from .catalog import Catalog
from .constants import (
    ID,
    IMG_TYPE,
    IR_069,
    IR_107,
    VERTICALLY_INTEGRATED_LIQUID,
    ImageType,
)
from .h5 import H5Store

logging.getLogger().setLevel(logging.INFO)

idx = pd.IndexSlice


class SEVIRGenerator(IterableDataset[tuple[Tensor, Tensor]]):
    __slots__ = ("reader", "img_ids", "x_img_types", "y_img_types", "x", "y")
    if typing.TYPE_CHECKING:
        reader: H5Store
        img_ids: list[str]
        x_img_types: set[ImageType]
        y_img_types: set[ImageType]
        x: Catalog
        y: Catalog

    def __init__(
        self,
        sevir: Catalog | str,
        *,
        inputs: set[ImageType],
        features: set[ImageType],
        validate: bool = True,
        catalog: str | None = None,
        data_dir: str | None = None,
        nproc: int | None = os.cpu_count(),
    ) -> None:
        super().__init__()
        self.meta = meta = (
            sevir
            if isinstance(sevir, Catalog)
            else Catalog(sevir, catalog=catalog, data_dir=data_dir, img_types=inputs | features)
        )

        self.x, self.y = x, y = meta.split_by_types(list(inputs), list(features))

        if validate:
            meta.validate()
            assert (
                len(meta.get_level_values(ID).unique())
                == len(x.get_level_values(ID).unique())
                == len(y.get_level_values(ID).unique())
            )
            assert (meta.img_types) == (inputs | features) == (x.img_types | y.img_types)

        self.reader = reader = FileReader(meta, nproc=nproc)
        self.img_ids = list(reader.index.get_level_values(ID).unique())

    @typing.overload
    def get_batch(
        self, n: int, img_id: str | int | None = ..., metadata: typing.Literal[False] = ...
    ) -> tuple[Tensor, Tensor]:
        ...

    @typing.overload
    def get_batch(
        self, n: int, img_id: str | int | None = ..., metadata: typing.Literal[True] = ...
    ) -> tuple[tuple[Tensor, Tensor], pd.DataFrame]:
        ...

    def get_batch(
        self, n: int, img_id: str | int | None = None, metadata: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], pd.DataFrame]:
        if img_id is None:
            img_id = random.choice(self.img_ids)
        elif isinstance(img_id, int):
            img_id = self.img_ids[img_id]

        # access reader with -> [(id, im_t), (file_index, l, w, t)] -> ndarray[img_type, l, w, t]
        x = np.array(
            [
                self.reader[idx[img_id, img_type], idx[self.x.file_index.loc[(img_id, img_type)], :, :, :]]
                for img_type in self.x.img_types
            ]
        )
        y = np.array([self.reader[idx[img_id, img_type], idx[n, :, :, :]] for img_type in self.y.img_types])
        values = torch.from_numpy(x), torch.from_numpy(y)
        if not metadata:
            return values
        return (values, self.meta[[img_id]])

    def close(self) -> None:
        logging.info("Closing SEVIRReaderHDF5")
        self.reader.close_all()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_batch(index)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        try:
            for img_id in self.img_ids:
                n = self.reader.n_event[img_id]
                logging.info(f"Loading {n} frames for event {img_id}")
                bar = tqdm.tqdm(total=n)
                for i in range(n):
                    yield self.get_batch(i, img_id)
                    bar.update(1)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            self.close()
            raise StopIteration

    def __enter__(self) -> typing.Generator[SEVIRGenerator, None, None]:
        try:
            yield self
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            self.close()
        finally:
            self.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        first, last = self.reader.index.get_level_values(ID).unique()[0:-1]
        ids = f"[{first!r}, ..., {last!r}]"

        shape = self.reader.shapes
        img_types = self.reader.index.get_level_values(IMG_TYPE).unique().tolist()

        n, l, w, t = shape.min()

        return f"{self.__class__.__name__}[idx[{ids}, {img_types!r}], [0:{n}, 0:{l}, 0:{w}, 0:{t}]]"


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
    features={VERTICALLY_INTEGRATED_LIQUID},
    validate: bool = True,
) -> None:
    generator = SEVIRGenerator(
        sevir,
        catalog=catalog,
        data_dir=data_dir,
        #
        inputs=inputs,
        features=features,
        validate=validate,
    )
    # x = generator.reader._store._index.dropna().reset_index().to_numpy()[:5, :3]
    # print(x, generator.reader._store[x])

    for (x,), s in generator.reader.groupby([ID]):
        print(x, s, sep="\n")
        break
    # print(
    #     generator.reader.pick(generator.reader.index[0]),
    #     generator.reader.select(generator.reader.index[0:5]),
    # )
    # for x, y in generator:
    #     print(x.shape, y.shape)
    # #     break
    # # for x, y in SEVIRLoader(generator, batch_size=15, num_workers=0):
    #     print(x.shape, y.shape)
    #     break
    generator.close()


if __name__ == "__main__":
    main()
