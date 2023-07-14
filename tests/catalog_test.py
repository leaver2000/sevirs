import pytest
import pandas as pd

from src.catalog import Catalog
from src.constants import WIDTH_M, HEIGHT_M, SIZE_X, SIZE_Y, EVENT_INDEX, IMG_TYPE


@pytest.fixture(scope="session")
def cat() -> Catalog:
    return Catalog()


def test_catalog_columns(cat: Catalog) -> None:
    assert isinstance(cat, Catalog)
    cat.validate_paths()
    df = cat.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert all(df[[WIDTH_M, HEIGHT_M]] == 384000.0)
    assert all(df[SIZE_X] == df[SIZE_Y])


def test_catalog_attributes(cat: Catalog) -> None:
    assert isinstance(cat, Catalog)
    assert isinstance(cat.file_names, pd.Series)
    assert cat.file_names.dtype == "string"
    assert isinstance(cat.index, pd.MultiIndex)
    assert cat.index.names == [EVENT_INDEX, IMG_TYPE]


def test_catalog_getitem(cat: Catalog) -> None:
    x0 = cat[0]
    assert isinstance(x0, pd.DataFrame)
    assert isinstance(x0.index, pd.Index)
    assert x0.index.name == IMG_TYPE

    x1 = cat[0:1]
    assert isinstance(x1, pd.DataFrame)
    assert isinstance(x1.index, pd.MultiIndex)
    assert x1.index.names == [EVENT_INDEX, IMG_TYPE]
