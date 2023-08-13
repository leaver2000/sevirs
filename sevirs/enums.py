from __future__ import annotations

import abc
from enum import Enum
from enum import EnumMeta as _EnumMeta
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike

from ._typing import EnumProtocol, ImageConfig


class PlotEnumProtocol(EnumProtocol, Protocol):
    @classmethod
    def gcf(cls) -> Figure:
        ...

    @classmethod
    def sequential(cls: type[EnumT], __iterable: Iterable[Any] | None = None) -> tuple[EnumT, ...]:
        ...


EnumT = TypeVar("EnumT", bound=PlotEnumProtocol)


class BaseEnumType(_EnumMeta):
    def sequential(self, __iterable: Iterable[Any] | None = None) -> tuple[EnumT, ...]:
        if not __iterable:
            return tuple(iter(self))
        if isinstance(__iterable, tuple) and all(
            isinstance(x, self) for x in __iterable
        ):
            return __iterable

        return tuple(self.__call__(x) for x in __iterable)


class PlotEnumType(BaseEnumType):
    def gcf(self) -> Figure:
        return plt.gcf() if plt.fignum_exists(1) else self.figure()

    def figure(self, figsize=(15, 5)) -> Figure:
        return plt.figure(figsize=figsize)

    def zipplots(self, nrows: int = 1, *args: EnumT | str) -> zip[tuple[EnumT, Axes]]:
        ncols = len(self) if not args else len(args)
        members = self.sequential(args)

        fig = self.gcf()
        axes = cast(np.ndarray, fig.subplots(nrows, ncols)).flatten()
        return zip((members * nrows), axes)


class ImageEnumType(str, Enum, metaclass=PlotEnumType):
    # TODO: a generic type annotated enum similar to pydantic

    def __str__(self) -> str:
        return str(self._value_)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    # =================================================================================================================
    # - metadata interface
    @property
    @abc.abstractmethod
    def metadata(self) -> Mapping[str, Mapping[str, Any]]:
        ...

    @property
    def description(self) -> str:
        return self.metadata[self.value]["description"]

    @property
    def sensor(self) -> str:
        return self.metadata[self.value]["sensor"]

    @property
    def patch_size(self) -> int:
        return self.metadata[self.value]["patchSize"]

    @property
    def time_steps(self) -> int:
        return self.metadata[self.value]["timeStep"]

    @property
    def boundaries(self) -> list[float] | dict[str, float]:
        return self.metadata[self.value]["boundaries"]

    @property
    def encoding(self) -> float:
        return self.metadata[self.value]["encoding"]

    @property
    def colors(self) -> list[list[float]] | str:
        return self.metadata[self.value]["colorMap"]

    # =================================================================================================================
    # a custom colormap
    @abc.abstractmethod
    def get_cmap(self) -> Colormap:
        ...

    # =================================================================================================================
    # - display interface

    def get_norm(self, *, encoded=True, ncolors: int = 0, clip=False) -> Normalize:
        bounds = self.boundaries
        if isinstance(bounds, dict):
            return Normalize(**{key: value * self.encoding for key, value in bounds.items()}, clip=clip)
        arr = np.array(bounds, dtype=np.float_)
        if encoded:
            arr *= self.encoding
        return BoundaryNorm(arr, ncolors=ncolors, clip=clip)

    def imconfig(
        self, *, encoded: bool = True, **kwargs: str | float | Sequence[float] | np.ndarray | None
    ) -> ImageConfig:
        config: ImageConfig = {"cmap": self.get_cmap()}
        config["norm"] = self.get_norm(encoded=encoded, ncolors=config["cmap"].N)
        config |= {key: value for key, value in kwargs.items() if value is not None}  # type: ignore
        return config

    def imshow(
        self,
        data: ArrayLike,
        axes: Axes | None = None,
        *,
        encoded: bool = True,
        title: str | None = None,
        # - ImageShowConfig
        interpolation: str | None = None,
        alpha: float | np.ndarray | None = None,
        extent: Sequence[float] | None = None,
        interpolation_stage: Literal["data", "rgba"] | None = None,
        filternorm: bool = True,
        filterrad: float = 4.0,
        resample: bool | None = None,
        url: str | None = None,
    ) -> AxesImage:
        config = self.imconfig(
            encoded=encoded,
            # cmap=cmap,
            # norm=norm,
            # aspect=aspect,
            interpolation=interpolation,
            alpha=alpha,
            # vmin=vmin,
            # vmax=vmax,
            # origin=origin,
            extent=extent,
            interpolation_stage=interpolation_stage,
            filternorm=filternorm,
            filterrad=filterrad,
            resample=resample,
            url=url,
        )
        if axes is None:
            axes = plt.gca()
            if not isinstance(axes, plt.Axes):
                raise TypeError("No Axes found")

        im = axes.imshow(data, **config)
        if title is not None:
            axes.set_title(title)
        return im

    def colorbar(self, ax: Axes | None = None) -> Colorbar:
        ax = ax or plt.gca()
        ax.set_title(self.value)
        cmap = self.get_cmap()
        return Colorbar(ax, cmap=cmap, orientation="vertical")
