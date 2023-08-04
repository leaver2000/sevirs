from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    overload,
)

import polars as pl
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler

from .catalog import Catalog
from .constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_PATH_TO_SEVIR,
    IMG_TYPE,
    ImageType,
)
from .generic import AbstractContextManager
from .h5 import H5Store

TensorPair = tuple[Tensor, Tensor]


class TensorGenerator(IterableDataset[TensorPair], AbstractContextManager["TensorGenerator"]):
    if TYPE_CHECKING:
        x: H5Store
        x_img_types: list[ImageType]

        y: H5Store
        y_img_types: list[ImageType]

        img_ids: list[str]
        patch_size: int

    def __init__(
        self,
        data: Catalog | str = DEFAULT_PATH_TO_SEVIR,
        *,
        inputs: tuple[ImageType, ...],
        features: tuple[ImageType, ...],
        catalog: str = DEFAULT_CATALOG,
        data_dir: str = DEFAULT_DATA,
        patch_size: int | Literal["up", "down"] = "up",
    ) -> None:
        super().__init__()

        img_types = inputs + features

        if len(set(img_types)) != len(img_types):
            raise ValueError("inputs and features must be unique!")

        data = (
            data
            if isinstance(data, Catalog)
            else Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir)
        )

        # image_ids are used by the iterator to get batches
        self.img_ids = data.id.unique().to_list()

        # the patch size is used to interpolate the so all the images fit.
        if isinstance(patch_size, int):
            self.patch_size = patch_size
        else:
            p_sizes = (img_type.patch_size for img_type in img_types)
            self.patch_size = max(p_sizes) if patch_size == "up" else min(p_sizes)

        # the catalog is split into inputs and features
        self.x, self.y = x, y = [H5Store(cat) for cat in data.split_by_types(list(inputs), list(features))]

        assert data.id.n_unique() == x.id.n_unique() == y.id.n_unique()
        assert len(img_types) == (len(x.types) + len(y.types))

    # =================================================================================================================
    @overload  # type: ignore[misc]
    def select(self, img_id: str, *, metadata: Literal[False] = False) -> TensorPair:
        ...

    @overload
    def select(self, img_id: str, *, metadata: Literal[True] = True) -> tuple[TensorPair, pl.DataFrame]:
        ...

    def select(self, img_id: str, *, metadata: bool = False) -> TensorPair | tuple[TensorPair, pl.DataFrame]:
        x = self.x.interpstack(img_id, self.patch_size)
        y = self.y.interpstack(img_id, self.patch_size)
        tensors = torch.from_numpy(x), torch.from_numpy(y)
        if metadata is True:
            return tensors, self.get_metadata(img_id)
        return tensors

    def __getitem__(self, index: int) -> TensorPair:
        return self.select(self.img_ids[index])

    # =================================================================================================================
    @overload  # type: ignore[misc]
    def iterate(self, *, metadata: Literal[False] = False) -> Iterable[TensorPair]:
        ...

    @overload
    def iterate(self, *, metadata: Literal[True] = True) -> Iterable[tuple[TensorPair, pl.DataFrame]]:
        ...

    def iterate(self, *, metadata: bool = False) -> Iterable[TensorPair | tuple[TensorPair, pl.DataFrame]]:
        logging.info("⛈️ Beginning Tensor Generation ⛈️")
        for img_id in tqdm.tqdm(self.img_ids):
            yield self.select(img_id, metadata=metadata)  # type: ignore[call-overload]

    def __iter__(self) -> Iterator[TensorPair]:
        yield from self.iterate(metadata=False)

    # =================================================================================================================
    def close(self) -> None:
        logging.info("🏪 Closing Store 🏪")
        self.x.close()
        self.y.close()

    def get_metadata(self, img_id: str) -> pl.DataFrame:
        frames = [
            self.x.data.filter(self.x.id == img_id),
            self.y.data.filter(self.y.id == img_id),
        ]
        return pl.concat(frames).sort(
            pl.col(IMG_TYPE).map_dict({v: i for i, v in enumerate(self.x.types + self.y.types)}),
        )


class TensorLoader(DataLoader[TensorPair]):
    if TYPE_CHECKING:
        dataset: TensorGenerator
        __iter__: Callable[[], Iterator[TensorPair]]  # type: ignore

    def __init__(
        self,
        dataset: TensorGenerator,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[TensorPair]], Any] | None = None,
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
