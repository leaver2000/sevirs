from sevirs.constants import FILE_INDEX, FILE_REF, ID, IMG_TYPE
from sevirs.core.h5 import Store


def test_store(store: Store) -> None:
    assert len(store._files) == len(store)
    assert {ID, FILE_INDEX, FILE_REF, IMG_TYPE}.issubset(store.catalog.columns)
