from __future__ import annotations

import abc
import itertools
import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Final,
    Generic,
    Hashable,
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

from .._typing import CatalogData, PatchSize
from ..constants import (
    DEFAULT_CATALOG,
    DEFAULT_DATA,
    DEFAULT_N_FRAMES,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATH_TO_SEVIR,
    IMG_TYPE,
    ImageType,
)
from ..generic import AbstractContextManager
from .catalog import Catalog
from .h5 import Store

ValueT = TypeVar("ValueT")
IndexT = TypeVar("IndexT", bound=Hashable)
DatasetT = TypeVar("DatasetT", "FeatureGenerator", "TimeSeriesGenerator")
TensorPair = tuple[Tensor, Tensor]


class SequentialGenerator(IterableDataset[ValueT], AbstractContextManager, Generic[IndexT, ValueT]):
    # metadata:Mapping[IndexT, pl.DataFrame]
    def __init__(
        self, indices: Iterable[IndexT], img_types: Sequence[ImageType], patch_size: PatchSize = DEFAULT_PATCH_SIZE
    ) -> None:
        super().__init__()
        # indices
        self.indices: Final[tuple[IndexT, ...]] = tuple(indices)
        assert len(self.indices) == len(set(self.indices)), "Indices must be unique"

        # patch_size
        if not isinstance(patch_size, int):
            p_sizes = (img_type.patch_size for img_type in img_types)
            patch_size = max(p_sizes) if patch_size == "upscale" else min(p_sizes)
        self.patch_size: Final[int] = patch_size

    # =================================================================================================================
    # - abstract methods
    @abc.abstractmethod
    def pick(self, index: IndexT) -> ValueT:
        ...

    @abc.abstractmethod
    def get_metadata(self, img_id: IndexT | None = None) -> pl.DataFrame:
        ...

    # =================================================================================================================
    # - overloads
    @overload  # type: ignore[misc]
    def select(self, index: IndexT, *, metadata: Literal[False] = False) -> ValueT:
        ...

    @overload
    def select(self, index: IndexT, *, metadata: Literal[True] = True) -> tuple[ValueT, pl.DataFrame]:
        ...

    def select(self, index: IndexT, *, metadata: bool = False) -> ValueT | tuple[ValueT, pl.DataFrame]:
        values = self.pick(index)
        if metadata:
            return values, self.get_metadata(index)
        return values

    @overload  # type: ignore[misc]
    def iterate(self, *, metadata: Literal[False] = False) -> Iterable[ValueT]:
        ...

    @overload
    def iterate(self, *, metadata: Literal[True] = True) -> Iterable[tuple[ValueT, pl.DataFrame]]:
        ...

    def iterate(self, *, metadata: bool = False) -> Iterable[ValueT | tuple[ValueT, pl.DataFrame]]:
        logging.info("🏃 Iterating over Dataset 🏃")
        for index in tqdm.tqdm(self.indices):
            yield self.select(index, metadata=metadata)  # type: ignore[call-overload]

    # =================================================================================================================
    # - dunder methods
    def __getitem__(self, idx: int) -> ValueT:
        return self.pick(self.indices[idx])

    def __iter__(self) -> Iterator[ValueT]:
        yield from self.iterate(metadata=False)

    def __len__(self) -> int:
        return len(self.indices)

    # =================================================================================================================
    # - not implemented
    def groupby(self, by: str | IntoExpr) -> Iterable[Sequence[ValueT]]:
        raise NotImplementedError


# =====================================================================================================================
class FeatureGenerator(SequentialGenerator[str, TensorPair]):
    def __init__(
        self,
        data: CatalogData = DEFAULT_PATH_TO_SEVIR,
        *,
        inputs: tuple[ImageType, ...],
        targets: tuple[ImageType, ...],
        catalog: str | None = DEFAULT_CATALOG,
        data_dir: str | None = DEFAULT_DATA,
        patch_size: PatchSize = DEFAULT_PATCH_SIZE,
        maintain_order: bool = False,
    ) -> None:
        img_types = inputs + targets

        if len(set(img_types)) != len(img_types):
            raise ValueError("inputs and targets must be unique!")

        if not isinstance(data, Catalog):
            data = Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir)

        # - store
        x, y = [Store(cat) for cat in data.intersect(inputs, targets)]
        assert set(x.id) == set(y.id) and len(img_types) == (len(x.types) + len(y.types))

        # - indices
        indices = x.id.unique(maintain_order=maintain_order)

        super().__init__(indices, img_types, patch_size)
        self.x: Final[Store] = x
        self.y: Final[Store] = y

    # =================================================================================================================
    # - abstract method interface
    def pick(self, index: Annotated[str, "The `img_id` for the event."]) -> TensorPair:
        x = self.x.interp(index, patch_size=self.patch_size)
        y = self.y.interp(index, patch_size=self.patch_size)
        return torch.from_numpy(x), torch.from_numpy(y)

    def get_metadata(self, index: str | None = None) -> pl.DataFrame:
        if index is None:
            frames = [self.x.data, self.y.data]
        else:
            frames = [
                self.x.data.filter(self.x.id == index),
                self.y.data.filter(self.y.id == index),
            ]
        return pl.concat(frames).sort(
            pl.col(IMG_TYPE).map_dict({v: i for i, v in enumerate(self.x.types + self.y.types)}),
        )

    def close(self) -> None:
        logging.info("🏪 Closing Store 🏪")
        self.x.close()
        self.y.close()


class TimeSeriesGenerator(SequentialGenerator[tuple[str, int], TensorPair]):
    def __init__(
        self,
        data: CatalogData = DEFAULT_PATH_TO_SEVIR,
        *,
        img_types: tuple[ImageType, ...],
        catalog: str | None = DEFAULT_CATALOG,
        data_dir: str | None = DEFAULT_DATA,
        patch_size: PatchSize = DEFAULT_PATCH_SIZE,
        n_frames: int = DEFAULT_N_FRAMES,
        n_inputs: int = 5,
        n_targets: int = 5,
        maintain_order: bool = False,
    ) -> None:
        if not isinstance(data, Catalog):
            data = Catalog(data, img_types=img_types, catalog=catalog, data_dir=data_dir)

        # - store
        store = Store(data)

        # - indices
        indices: itertools.product[tuple[str, int]] = itertools.product(
            store.id.unique(maintain_order=maintain_order),
            (i for i in range(0, n_frames, n_inputs) if i + n_inputs + n_targets <= n_frames),
        )
        super().__init__(indices, img_types, patch_size)
        self.store = store

        # time slice parameters
        self.n_frames = n_frames
        self.n_inputs = n_inputs
        self.n_targets = n_targets

    # =================================================================================================================
    # - abstract method interface
    def pick(self, index: tuple[str, int]) -> TensorPair:
        img_id, x_start = index
        x_stop = x_start + self.n_inputs
        y_start = x_stop + 1
        y_stop = y_start + self.n_targets

        arr = self.store.interp(img_id, patch_size=self.patch_size)

        x = arr[..., x_start:x_stop]
        y = arr[..., y_start:y_stop]
        return (torch.from_numpy(x), torch.from_numpy(y))

    def get_metadata(self, index: tuple[str, int] | None = None) -> pl.DataFrame:
        raise NotImplementedError
        if index is None:
            return self.store.data
        img_id, stop = index
        return self.store.data.filter((self.store.id == img_id))  # & (self.store.time == self.time_idx.index(stop)),

    def close(self) -> None:
        self.store.close()


class TensorLoader(DataLoader[TensorPair], AbstractContextManager, Generic[DatasetT]):
    if TYPE_CHECKING:
        dataset: DatasetT
        __iter__: Callable[..., Iterator[TensorPair]]  # type: ignore

    def __init__(
        self,
        dataset: DatasetT,
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

    def close(self) -> None:
        self.dataset.close()
