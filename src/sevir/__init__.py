__all__ = [
    "Catalog",
    "TensorGenerator",
    "TensorLoader",
    "H5File",
    "Store",
    "ImageType",
    "constants",
    "plot",
]
from . import constants
from .constants import ImageType
from .core.catalog import Catalog
from .core.datasets import TensorGenerator, TensorLoader
from .core.h5 import H5File, Store

try:
    import cartopy.crs  # noqa: F401

    from .core import plot
except ImportError:

    class _dummy:
        def __getattr__(self, item):
            raise ImportError(f"Could not import {item} from cartopy.crs")

    plot = _dummy()  # type: ignore[assignment]
