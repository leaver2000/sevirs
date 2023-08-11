import os

os.environ["PATH_TO_SEVIR"] = "/mnt/data/sevir"

import pytest

from sevirs.core.catalog import Catalog
from sevirs.core.h5 import Store


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    return Catalog()


@pytest.fixture(scope="session")
def store(catalog: Catalog) -> Store:
    return Store(catalog)
