# pyright: reportGeneralTypeIssues=false
"""
Creates plots of SEVIR events using cartopy library
"""
from __future__ import annotations

import re
from typing import Final, Iterable, Mapping, TypeAlias

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.figure
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cartopy.crs import Globe
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from .._typing import DictStrAny
from ..constants import (
    CONFIG,
    ID,
    IR_069,
    IR_107,
    LGHT,
    LL_LAT,
    LL_LON,
    PROJ,
    UR_LAT,
    UR_LON,
    VIL,
    VIS,
    ImageType,
)

NormalizedColorMap: TypeAlias = "dict[str, mpl.colors.ListedColormap | mpl.colors.BoundaryNorm]"


# =====================================================================================================================
def zoombox(ax: Axes | None = None, *, x: int, y: int, size: int = 10) -> tuple[int, int]:
    x0, y0 = x - size, y - size
    x1, y1 = x + size, y + size

    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("ax must be an Axes instance")
    args = [x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0]
    ax.plot(*args, "-k", lw=3)
    ax.plot(*args, "-", color="dodgerblue", lw=2)

    return x, y


def geoaxes(
    *args: int, fig: matplotlib.figure.Figure | None = None, projection: ccrs.Projection = ccrs.PlateCarree()
) -> tuple[matplotlib.figure.Figure, GeoAxes]:
    assert ccrs is not None
    fig = fig or plt.gcf()
    ax = fig.add_subplot(*args, projection=projection)
    return fig, ax


def subplots(
    nrows: int = 1, ncols: int = 1, *, figsize: tuple[int, ...] | None = None, ratio: int = 4
) -> tuple[matplotlib.figure.Figure, Iterable[tuple[plt.Axes, ...]]]:
    if figsize is None:
        figsize = (ncols * ratio, nrows * ratio)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    return (fig, axs)


def timegrid(
    x: int, xs: int, y: int, ratio: int = 4
) -> tuple[plt.Figure, Mapping[tuple[int, int], plt.Axes], list[tuple[int, int]]]:
    """time series plots"""
    fig, axs = plt.subplots(y, x, figsize=(x * ratio, y * ratio))
    indices = [(i, i * (xs // x)) for i in range(x)]
    return (fig, axs, indices)


def values(values: np.ndarray, ax: Axes | None = None) -> Axes | GeoAxes:
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, Axes):
        raise TypeError("could not resolve Axes instance")
    assert len(values.shape) == 2

    x = np.arange(0, values.shape[0])
    y = np.arange(0, values.shape[1])
    X, Y = np.meshgrid(x, y)
    X = np.ravel(X)
    Y = np.ravel(Y)
    V = np.ravel(values)
    for i in np.arange(0, len(X)):
        fillstr = np.asarray(np.round(V[i], 2), dtype=str)
        fillstr = np.char.ljust(fillstr, 4, "0")
        if np.round(V[i], 2) > 0.5:
            ax.text(X[i] - 0.2, Y[i], fillstr, color="k")
        else:
            ax.text(X[i] - 0.2, Y[i], fillstr, color="w")
    return ax


# =====================================================================================================================
def get_colors_and_boundaries(img_type: ImageType) -> tuple[np.ndarray, np.ndarray]:
    return np.array(CONFIG["display"][img_type]["colors"]), np.array(CONFIG["display"][img_type]["boundaries"])


def get_cmap(colors: np.ndarray, bad: int, under: int, over: int, good: slice) -> ListedColormap:
    cmap = ListedColormap(colors[good])
    cmap.set_bad(colors[bad])
    cmap.set_under(colors[under])
    cmap.set_over(colors[over])
    return cmap


def get_crs(df: pl.DataFrame) -> tuple[ccrs.LambertAzimuthalEqualArea, tuple[float, float, float, float]]:
    """
    Gets cartopy coordinate reference system and image extent for SEVIR events

    Parameters
    ---------
    info pandas.Series object
        Any row from the SEVIR CATALOG, or metadata returned from SEVIRGenerator

    Returns
    -------
    ccrs - cartopy.crs object containing coordinate ref system
    img_extent - image_extent used for imshow
    """

    if not df[ID].n_unique() == 1:
        raise ValueError("info must contain only one event")

    (data,) = df.select(PROJ, LL_LON, LL_LAT, UR_LAT, UR_LON).unique().to_dicts()
    data |= parse_projection(data[PROJ])

    globe = Globe(datum=data["datum"], ellipse=data["ellps"], semimajor_axis=data["a"], semiminor_axis=data["b"])
    if not (("proj" in data) and (data["proj"] == "laea")):
        raise ValueError("Only Lambert Azimuthal Equal Area projection is supported")

    crs = ccrs.LambertAzimuthalEqualArea(central_longitude=data["lon_0"], central_latitude=data["lat_0"], globe=globe)

    # use crs to compute image extent
    x1, y1 = crs.transform_point(data[LL_LON], data[LL_LAT], ccrs.Geodetic())
    x2, y2 = crs.transform_point(data[UR_LON], data[UR_LAT], ccrs.Geodetic())

    return crs, (x1, x2, y1, y2)


# =====================================================================================================================
def vil_cmap(encoded: bool = True) -> NormalizedColorMap:
    colors, bounds = get_colors_and_boundaries(VIL)

    if encoded:
        # TODO:  encoded:bool=False
        pass
    cmap = get_cmap(colors, 0, 1, -1, slice(1, -1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return {"cmap": cmap, "norm": norm}


def vis_cmap(encoded: bool = True) -> NormalizedColorMap:
    colors, bounds = get_colors_and_boundaries(VIS)

    if encoded:
        bounds *= 1e4
    cmap = get_cmap(colors, 0, 0, -1, slice(0, -1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return {"cmap": cmap, "norm": norm}


def ir_cmap(encoded: bool = True) -> NormalizedColorMap:
    colors, bounds = get_colors_and_boundaries(IR_069)

    if encoded:
        bounds *= 1e2
    cmap = get_cmap(colors, 0, 1, -1, slice(1, -1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return {"cmap": cmap, "norm": norm}


def c09_cmap(encoded: bool = True) -> dict[str, ListedColormap]:
    colors, _ = get_colors_and_boundaries(IR_069)
    return {"cmap": ListedColormap(colors)}


def with_kwargs(img_type: ImageType, *, encoded: bool = True) -> DictStrAny:
    if img_type == VIS:
        return vis_cmap(encoded)

    elif img_type == VIL:
        return vil_cmap(encoded)

    elif img_type == IR_069:
        vmin, vmax = (-8000, -1000) if encoded else (-80, -10)
        return c09_cmap(encoded) | {"vmin": vmin, "vmax": vmax}

    elif img_type == IR_107:
        vmin, vmax = (-7000, 2000) if encoded else (-70, 20)
        return {"cmap": "jet", "vmin": vmin, "vmax": vmax}

    elif img_type == LGHT:
        return {"cmap": "hot", "vmin": 0, "vmax": 5}
    raise ValueError(f"Invalid image type: {img_type}")


# =====================================================================================================================


# =====================================================================================================================
_DEFAULT_PROJECTION = {"a": None, "b": None, "lon_0": 0.0, "lat_0": 0.0, "ellps": "WGS84", "datum": None}


def parse_projection(proj: str) -> DictStrAny:
    data = _DEFAULT_PROJECTION.copy()
    for item in re.sub(r"\+", "", proj.strip()).split():
        k_v = item.split("=")
        if len(k_v) != 2:
            raise ValueError(f"Invalid projection string: {proj}, could not parse key-value pair")
        key, val = k_v
        data[key] = float(val) if key in ("lat_0", "lon_0", "a") else val
    return data


EarthFeature = tuple[cfeature.NaturalEarthFeature, DictStrAny]


_DEFAULT_NATURAL_EARTH_FEATURES: Final[tuple[EarthFeature, ...]] = (
    (cfeature.STATES, {}),
    (cfeature.LAKES, {"alpha": 0.5}),
    (cfeature.RIVERS, {}),
)
# ax.add_feature(cfeature.STATES)
# # ax.add_feature(cfeature.LAND)
# # ax.add_feature(cfeature.OCEAN)
# # ax.add_feature(cfeature.COASTLINE)
# # ax.add_feature(cfeature.BORDERS )
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.RIVERS)


def animation(
    frames: np.ndarray,
    meta: pl.DataFrame,
    img_type: ImageType,
    fig: matplotlib.figure.Figure = None,
    interval: int = 100,
    title: str | None = None,
    features: Iterable[EarthFeature] = _DEFAULT_NATURAL_EARTH_FEATURES,
) -> FuncAnimation:
    proj, (xll, xur, yll, yur) = get_crs(meta)

    fig, ax = geoaxes(1, 1, 1, fig=fig, projection=proj)

    ax.set_xlim((xll, xur))
    ax.set_ylim((yll, yur))
    # cmap, norm, vmin, vmax = get_cmap(img_type)
    im = ax.imshow(
        frames[:, :, 0],
        interpolation="nearest",
        origin="lower",
        extent=[xll, xur, yll, yur],
        transform=proj,
        **with_kwargs(img_type),
    )

    for feature, kwargs in features:
        ax.add_feature(feature, **kwargs)
    if title:
        ax.set_title(title)

    def animate(i):
        im.set_data(frames[:, :, i])
        return (im,)

    return FuncAnimation(
        fig,
        animate,
        init_func=lambda: [im],
        frames=range(frames.shape[2]),
        interval=interval,
        blit=True,
    )
