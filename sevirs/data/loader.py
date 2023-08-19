from __future__ import annotations

from typing import Any, Iterable, Literal, TypeVar

import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import GeneratorEnqueuer

from .generic import DataGenerator, Indicies, ThreadSafeIterator
from .sampler import BatchSampler, Sampler

T = TypeVar("T", bound=Any)  # input data type


class DataLoader(Iterable[T]):
    def __init__(
        self,
        data: DataGenerator[Any, T],
        /,
        *,
        sampler: Sampler[Any, Any]
        | Indicies[Any]
        | Literal["random", "philox", "threefry", "sequential"] = "sequential",
        batch_size: int = 2,
        num_samples: int | None = None,
        generator: tf.random.Generator | None = None,
        drop_last: bool = False,
        # - GeneratorEnqueuer Arguments
        timeout: float | None = None,
        workers: int = 1,
        max_queue_size: int = 10,
    ) -> None:
        super().__init__()
        # - Validation Logic
        if not isinstance(sampler, BatchSampler):
            sampler = (
                BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
                if not isinstance(sampler, str)
                else BatchSampler.create(data, sampler, num_samples, generator, batch_size, drop_last)
            )

        if workers <= 0:
            raise ValueError(f"Invalid number of workers: {workers}")

        # - Create Iterator
        it = ([data[idx] for idx in batch_idx] for batch_idx in sampler)

        # - Set Attributes
        self.enqueuer = GeneratorEnqueuer(it if workers == 1 else ThreadSafeIterator(it))
        self.timeout = timeout
        self.workers = workers
        self.max_queue_size = max_queue_size

    def __iter__(self) -> Iterable[list[T]]:
        self.enqueuer.start(
            workers=self.workers,
            max_queue_size=self.max_queue_size,
        )
        try:
            yield from self.enqueuer.get()
        finally:
            self.enqueuer.stop(self.timeout)
