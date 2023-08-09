from __future__ import annotations

import abc
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    overload,
)

import polars as pl
import torch
import tqdm
from polars.type_aliases import IntoExpr
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, Sampler

from .._typing import CatalogData
from ..constants import DEFAULT_PATH_TO_SEVIR, IMG_TYPE, ImageType
from ..generic import AbstractContextManager
from .catalog import Catalog
from .h5 import Store

T = TypeVar("T")


class AbstractDataset(IterableDataset[T], abc.ABC, Generic[T]):
    def __init__(self, img_ids: list[str]) -> None:
        self.img_ids: Final[list[str]] = img_ids
        super().__init__()

    @overload  # type: ignore[misc]
    @abc.abstractmethod
    def select(self, img_id: str, *, metadata: Literal[False] = False) -> T:
        ...

    @overload
    @abc.abstractmethod
    def select(self, img_id: str, *, metadata: Literal[True] = True) -> tuple[T, pl.DataFrame]:
        ...

    @abc.abstractmethod
    def select(self, img_id: str, *, metadata: bool = False) -> T | tuple[T, pl.DataFrame]:
        ...

    def __getitem__(self, index: int) -> T:
        id_ = self.img_ids[index]
        return self.select(id_)

    @abc.abstractmethod
    def close(self) -> None:
        ...

    # =================================================================================================================
    @abc.abstractmethod
    def get_metadata(self, img_id: str | None = None) -> pl.DataFrame:
        ...

    def group_by(self, by: str | IntoExpr) -> Iterable[Sequence[T]]:
        raise NotImplementedError

    # =================================================================================================================
    @overload  # type: ignore[misc]
    def iterate(self, *, metadata: Literal[False] = False) -> Iterable[T]:
        ...

    @overload
    def iterate(self, *, metadata: Literal[True] = True) -> Iterable[tuple[T, pl.DataFrame]]:
        ...

    def iterate(self, *, metadata: bool = False) -> Iterable[T | tuple[T, pl.DataFrame]]:
        logging.info("🏃 Iterating over Dataset 🏃")
        for img_id in tqdm.tqdm(self.img_ids):
            yield self.select(img_id, metadata=metadata)  # type: ignore[call-overload]

    def __iter__(self) -> Iterator[T]:
        yield from self.iterate(metadata=False)


# =====================================================================================================================
TensorPair = tuple[Tensor, Tensor]


class TensorGenerator(AbstractDataset[TensorPair], AbstractContextManager["TensorGenerator"]):
    if TYPE_CHECKING:
        x: Store
        y: Store
        patch_size: int

    def __init__(
        self,
        data: CatalogData = DEFAULT_PATH_TO_SEVIR,
        *,
        inputs: tuple[ImageType, ...],
        targets: tuple[ImageType, ...],
        catalog: str | None = None,
        data_dir: str | None = None,
        patch_size: int | Literal["upscale", "downscale"] = "upscale",
    ) -> None:
        img_types = inputs + targets

        if len(set(img_types)) != len(img_types):
            raise ValueError("inputs and targets must be unique!")

        if not isinstance(data, Catalog):
            data = Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir)
        # the patch size is used to interpolate the so all the images fit.
        if not isinstance(patch_size, int):
            p_sizes = (img_type.patch_size for img_type in img_types)
            patch_size = max(p_sizes) if patch_size == "upscale" else min(p_sizes)

        # the catalog is split into inputs and targets
        x, y = [Store(cat) for cat in data.intersect(inputs, targets)]
        assert set(x.id) == set(y.id)
        assert len(img_types) == (len(x.types) + len(y.types))

        super().__init__(x.id.unique().to_list())
        self.patch_size = patch_size
        self.x = x
        self.y = y

    # =================================================================================================================
    @overload  # type: ignore[misc]
    def select(self, img_id: str, *, metadata: Literal[False] = False) -> TensorPair:
        ...

    @overload
    def select(self, img_id: str, *, metadata: Literal[True] = True) -> tuple[TensorPair, pl.DataFrame]:
        ...

    def select(self, img_id: str, *, metadata: bool = False) -> TensorPair | tuple[TensorPair, pl.DataFrame]:
        x = self.x.interp_stack(img_id, patch_size=self.patch_size)
        y = self.y.interp_stack(img_id, patch_size=self.patch_size)
        tensors = torch.from_numpy(x), torch.from_numpy(y)
        if metadata is True:
            return tensors, self.get_metadata(img_id)
        return tensors

    # =================================================================================================================
    def close(self) -> None:
        logging.info("🏪 Closing Store 🏪")
        self.x.close()
        self.y.close()

    def get_metadata(self, img_id: str | None = None) -> pl.DataFrame:
        if img_id is None:
            frames = [self.x.data, self.y.data]
        else:
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
        __iter__: Callable[..., Iterator[TensorPair]]  # type: ignore

    def __init__(
        self,
        dataset: AbstractDataset,
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
