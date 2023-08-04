# pyright: reportGeneralTypeIssues=false
"""
Creates plots of SEVIR events using cartopy library
"""
import json
import os
import re
from typing import Any, Mapping, TypeAlias

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cartopy.crs import Globe
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from .constants import (
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

NormalizedColorMap: TypeAlias = dict[str, mpl.colors.ListedColormap | mpl.colors.BoundaryNorm]

# the previous color maps and boundaries we moved to a json file
with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
    _config = json.load(f)


# =====================================================================================================================
def get_colors_and_boundaries(img_type: ImageType) -> tuple[np.ndarray, np.ndarray]:
    return np.array(CONFIG["display"][img_type]["colors"]), np.array(CONFIG["display"][img_type]["boundaries"])


def get_cmap(colors: np.ndarray, bad: int, under: int, over: int, good: slice) -> ListedColormap:
    cmap = ListedColormap(colors[good])
    cmap.set_bad(colors[bad])
    cmap.set_under(colors[under])
    cmap.set_over(colors[over])
    return cmap


def with_kwargs(img_type: ImageType, *, encoded: bool = True) -> dict[str, Any]:
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


# =====================================================================================================================
def time_series_plots(
    x: int, xs: int, y: int, ratio: int = 4
) -> tuple[plt.Figure, Mapping[tuple[int, int], plt.Axes], list[tuple[int, int]]]:
    fig, axs = plt.subplots(y, x, figsize=(x * ratio, y * ratio))
    indices = [(i, i * (xs // x)) for i in range(x)]
    return (fig, axs, indices)


# =====================================================================================================================
_DEFAULT_PROJECTION = {"a": None, "b": None, "lon_0": 0.0, "lat_0": 0.0, "ellps": "WGS84", "datum": None}


def parse_projection(proj: str) -> dict[str, Any]:
    data = _DEFAULT_PROJECTION.copy()
    for p in re.sub(r"\+", "", proj.strip()).split():
        k_v = p.split("=")
        if len(k_v) != 2:
            raise ValueError(f"Invalid projection string: {proj}")

        key, val = k_v
        data[key] = float(val) if key in ("lat_0", "lon_0", "a") else val
    return data


def animation(
    frames: np.ndarray,
    meta: pl.DataFrame,
    img_type: ImageType,
    fig: matplotlib.figure.Figure = None,
    interval: int = 100,
    title: str | None = None,
) -> FuncAnimation:
    proj, img_extent = get_crs(meta)
    if fig is None:
        fig = plt.gcf()
    # return
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    xll, xur = img_extent[0], img_extent[1]
    yll, yur = img_extent[2], img_extent[3]
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

    ax.add_feature(cfeature.STATES)
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS )
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
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
