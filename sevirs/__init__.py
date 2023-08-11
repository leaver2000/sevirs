__all__ = [
    "Catalog",
    "TimeSeriesGenerator",
    "FeatureGenerator",
    "TensorLoader",
    "FileReader",
    "Store",
    "ImageType",
    "constants",
    "plot",
]
from . import constants
from .constants import ImageType
from .core.catalog import Catalog
from .core.datasets import FeatureGenerator, TensorLoader, TimeSeriesGenerator
from .core.h5 import FileReader, Store

try:
    import cartopy.crs  # noqa: F401

    from .core import plot
except ImportError:

    class _dummy:
        def __getattr__(self, item):
            raise ImportError(f"Could not import {item} from cartopy.crs")

    plot = _dummy()  # type: ignore[assignment]
