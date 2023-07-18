from sevir.catalog import Catalog
from sevir.h5 import H5Store
import polars as pl
from sevir.constants import ID, FILE_INDEX, DATA_INDEX, IMG_TYPE


def test_store(store: H5Store) -> None:
    assert len(store.data) == len(store)
    assert {ID, FILE_INDEX, DATA_INDEX, IMG_TYPE}.issubset(store.index.columns)
