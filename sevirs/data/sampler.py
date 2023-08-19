from __future__ import annotations

import abc
from typing import Any, Final, Generic, Iterator, Literal, TypeVar

import numpy as np
import tensorflow as tf

from .generic import Indicies

T = TypeVar("T", bound=Any)  # input data type
R = TypeVar("R")  # return data type data = iter(range(10))
ZERO: tf.Tensor = tf.constant(0, dtype=tf.int64)  # type: ignore[assignment]


class Sampler(Indicies[R], Generic[T, R], abc.ABC):
    __slots__ = ("indicies",)

    def __init__(self, indicies: Indicies[T], /) -> None:
        self.indicies: Final = indicies


class SequentialSampler(Sampler[T, int]):
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __len__(self) -> int:
        return len(self.indicies)


class RandomSampler(Sampler[T, int]):
    def __init__(
        self,
        indicies: Indicies[T],
        /,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
        generator: tf.random.Generator | None = None,
        alg: Literal["philox", "threefry"] | None = None,
    ) -> None:
        super().__init__(indicies)
        self.replacement = replacement
        if num_samples is not None and num_samples <= 0:
            raise ValueError("num_samples should be a positive integer value, ")
        self.num_samples = num_samples or len(self.indicies)
        self.generator = generator or tf.random.Generator.from_non_deterministic_state(alg)

    def __iter__(self) -> Iterator[int]:
        n = self.num_samples
        if self.replacement:
            for _ in range(self.num_samples):
                yield self.generator.uniform(shape=(), minval=0, maxval=n, dtype=tf.int64)
        else:
            yield from self.generator.uniform(shape=(n,), minval=0, maxval=n, dtype=tf.int64)

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[T, int]):
    def __init__(
        self,
        indices: Indicies[T],
        /,
        *,
        generator: tf.random.Generator | None = None,
        # generator: tf.random.Generator | None = None,
        alg: Literal["philox", "threefry"] | None = None,
    ) -> None:
        super().__init__(indices)
        self.generator = generator or tf.random.Generator.from_non_deterministic_state(alg)

    def __iter__(self) -> Iterator[int]:
        index = self.generator.uniform(
            shape=(len(self.indicies),), minval=0, maxval=len(self.indicies), dtype=tf.int64
        )
        return iter(index)

    def __len__(self) -> int:
        return len(self.indicies)


class WeightedRandomSampler(Sampler[float, int]):
    def __init__(
        self,
        weights: Indicies[float],
        /,
        *,
        num_samples: int,
        replacement: bool = True,
        generator: tf.random.Generator | None = None,
        alg: Literal["philox", "threefry"] | None = None,
    ) -> None:
        super().__init__(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator or tf.random.Generator.from_non_deterministic_state(alg)

    def __iter__(self) -> Iterator[int]:
        index = self.generator.uniform(shape=(self.num_samples,), minval=0, maxval=len(self.indicies), dtype=tf.int64)
        return iter(index)

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[T, list[int]]):
    def __init__(
        self,
        sampler: Indicies[T],
        /,
        *,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        super().__init__(sampler)
        self.batch_size = batch_size
        self.drop_last = drop_last

    @classmethod
    def create(
        cls,
        sampler: Indicies[int],
        strategy: Literal["random", "philox", "threefry", "sequential"] = "random",
        num_samples: int | None = None,
        generator: tf.random.Generator | None = None,
        batch_size: int = 2,
        drop_last: bool = False,
    ) -> BatchSampler[int]:
        if strategy in ("random", "philox", "threefry"):
            alg = strategy if strategy != "random" else None

            sampler = (
                SubsetRandomSampler(sampler, generator=generator, alg=alg)
                if num_samples is None
                else WeightedRandomSampler(sampler, num_samples=num_samples, generator=generator, alg=alg)
            )
        if strategy == "sequential":
            sampler = SequentialSampler(sampler)
        return cls(sampler, batch_size=batch_size, drop_last=drop_last)

    def __iter__(self) -> Iterator[list[T]]:
        if self.drop_last:
            sampler_iter = iter(self.indicies)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
            return
        batch = self.new_batch()
        idx_in_batch = 0
        for idx in self.indicies:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = self.new_batch()
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self) -> int:
        pad = 0 if self.drop_last else self.batch_size - 1
        return (len(self.indicies) + pad) // self.batch_size

    def new_batch(self) -> list[T]:
        return [ZERO] * self.batch_size  # type: ignore[return-value]


def main() -> None:
    NUM_SAMPLES = 5
    TOTAL_SAMPLES = 10
    data = tf.convert_to_tensor(np.random.randint(0, 100, TOTAL_SAMPLES))

    seq = SequentialSampler(range(TOTAL_SAMPLES))  # type: Indicies[int]
    assert list(seq) == list(range(TOTAL_SAMPLES))
    assert len(seq) == TOTAL_SAMPLES == len(list(iter(seq)))

    rs = RandomSampler(range(TOTAL_SAMPLES), num_samples=5)
    items = list(rs)
    assert len(items) == NUM_SAMPLES == len(rs)
    assert all(isinstance(x, tf.Tensor) for x in items)

    rs = RandomSampler(range(TOTAL_SAMPLES))
    items = list(rs)
    assert len(items) == TOTAL_SAMPLES == len(rs)
    assert all(isinstance(x, tf.Tensor) for x in items)

    wrs = WeightedRandomSampler(np.random.rand(TOTAL_SAMPLES), num_samples=NUM_SAMPLES)
    indicies = list(wrs)
    assert len(wrs) == NUM_SAMPLES == len(indicies)
    for idx in indicies:
        assert isinstance(data[idx], tf.Tensor)

    bs = BatchSampler(SequentialSampler(range(TOTAL_SAMPLES)), batch_size=2, drop_last=False)
    for batch in bs:
        assert isinstance(batch, list)
        assert len(batch) == 2
        x, y = batch
        assert all(isinstance(i, int) for i in (x, y))

    bs = BatchSampler(np.arange(TOTAL_SAMPLES, dtype=np.float32), batch_size=2, drop_last=False)
    for batch in bs:
        assert isinstance(batch, list)
        assert len(batch) == 2
        x, y = batch
        assert all(isinstance(i, np.floating) for i in batch)

    bs = BatchSampler(
        WeightedRandomSampler(np.random.rand(10_000), num_samples=70),
        batch_size=3,
        drop_last=True,
    )
    for i, batch in enumerate(bs):
        assert isinstance(batch, list)
        assert len(batch) == 3
        x, y, z = batch
        print(i)
        print(x, y, z, sep="\n")
        print("======================")

    bs = BatchSampler.create(
        np.arange(TOTAL_SAMPLES, dtype=np.float32),
        strategy="random",
    )


if __name__ == "__main__":
    main()
