from __future__ import annotations

import contextlib
import logging
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    overload,
)

import polars as pl
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler

from .catalog import Catalog
from .constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_PATH_TO_SEVIR,
    IR_069,
    IR_107,
    LGHT,
    VIL,
    VIS,
    ImageType,
)
from .generic import AbstractContextManager
from .h5 import H5Store

logging.getLogger().setLevel(logging.INFO)


class TensorGenerator(IterableDataset[tuple[Tensor, Tensor]], AbstractContextManager["TensorGenerator"]):
    __slots__ = ("image_ids", "x", "y", "x_img_types", "y_img_types")
    if TYPE_CHECKING:
        image_ids: Final[list[str]]
        x_img_types: Final[list[ImageType]]
        y_img_types: Final[list[ImageType]]
        x: Final[H5Store]
        y: Final[H5Store]

    def __init__(
        self,
        data: Catalog | str = DEFAULT_PATH_TO_SEVIR,
        *,
        inputs: tuple[ImageType, ...],
        features: tuple[ImageType, ...],
        catalog: str = DEFAULT_CATALOG,
        data_dir: str = DEFAULT_DATA,
    ) -> None:
        super().__init__()

        img_types = inputs + features
        if len(set(img_types)) != len(img_types):
            raise ValueError("inputs and features must be unique")
        data = (
            data
            if isinstance(data, Catalog)
            else Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir)
        )

        self.image_ids = data.id.unique().to_list()
        self.x, self.y = x, y = [H5Store(cat) for cat in data.split_by_types(list(inputs), list(features))]

        assert data.id.n_unique() == x.catalog.id.n_unique() == y.catalog.id.n_unique()
        assert len(img_types) == (len(x.catalog.types) + len(y.catalog.types))
        self.x_img_types, self.y_img_types = list(x.catalog.types), list(y.catalog.types)

    @overload  # type: ignore[misc]
    def get_batch(
        self,
        img_id=...,
        img_type=...,
        metadata: Literal[False] = ...,
    ) -> tuple[Tensor, Tensor]:
        ...

    @overload
    def get_batch(
        self,
        img_id=...,
        img_type=...,
        metadata: Literal[True] = ...,
    ) -> tuple[tuple[Tensor, Tensor], pl.DataFrame]:
        ...

    def get_batch(
        self,
        img_id: int | str | bytes | None = None,
        img_type: list[ImageType] | None = None,
        metadata: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], pl.DataFrame]:
        #
        if img_id is None or isinstance(img_id, int):
            img_id = self.image_ids[img_id] if img_id is not None else random.choice(self.image_ids)

        x = self.x.get_batch(img_id, img_types=self.x_img_types)  # [img_id, img_type or self.x_img_types])
        y = self.y.get_batch(img_id, img_types=self.y_img_types)
        # return x, y
        values = x, y  # torch.from_numpy(x), torch.from_numpy(y)
        if metadata is True:
            return (values, self.x.catalog.data.filter(self.x.catalog.id == img_id))
        return values

    @overload  # type: ignore[misc]
    def iter_batches(self, *, metadata: Literal[False] = ...) -> Iterator[tuple[Tensor, Tensor]]:
        ...

    @overload
    def iter_batches(self, *, metadata: Literal[True] = ...) -> Iterator[tuple[tuple[Tensor, Tensor], pl.DataFrame]]:
        ...

    def iter_batches(
        self, *, metadata: bool = False
    ) -> Iterator[tuple[Tensor, Tensor]] | Iterator[tuple[tuple[Tensor, Tensor], pl.DataFrame]]:
        bar = tqdm.tqdm(total=len(self.image_ids))
        for img_id in self.image_ids:
            yield self.get_batch(img_id, metadata=metadata)  # type: ignore[call-overload]
            bar.update(1)
        bar.close()

    def close(self) -> None:
        logging.info("Closing Store")
        self.x.close()
        self.y.close()

    @contextlib.contextmanager
    def session(self):
        try:
            yield self
        finally:
            self.close()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_batch(index)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        # try:
        for img_id in self.image_ids:
            yield self.get_batch(img_id)

        # except KeyboardInterrupt:
        #     logging.info("KeyboardInterrupt")
        #     self.close()
        #     raise StopIteration


class TensorLoader(DataLoader[tuple[Tensor, Tensor]]):
    if TYPE_CHECKING:
        dataset: TensorGenerator

    def __init__(
        self,
        dataset: TensorGenerator,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[tuple[Tensor, Tensor]]], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
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

    def __enter__(self) -> TensorLoader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.dataset.close()


def main(
    sevir="/mnt/data/c/sevir",
    data_dir: str = "data",
    catalog: str = "CATALOG.csv",
    inputs=(IR_069, IR_107),
    features=(VIL,),
) -> None:
    logging.info(
        f"""\
⛈️ TensorGenerator Example ⛈️
{VIL=}
{IR_069=}
{IR_107=}
{VIS=}
{LGHT=}"""
    )

    with TensorGenerator(
        sevir, catalog=catalog, data_dir=data_dir, inputs=inputs, features=features
    ).session() as generator:
        i = 0
        for (l, r), df in generator.iter_batches(metadata=True):
            print(df, l, r, sep="\n")
            if i > 10:
                break
            i += 1


if __name__ == "__main__":
    main()
