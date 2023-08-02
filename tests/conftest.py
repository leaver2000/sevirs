import os

os.environ["PATH_TO_SEVIR"] = "/mnt/data/c/sevir"

import pytest
from sevir.catalog import Catalog
from sevir.h5 import H5Store


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog()


@pytest.fixture(scope="session")
def store(catalog: Catalog) -> H5Store:
    return H5Store(catalog)
