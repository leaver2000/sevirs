from sevir.h5 import H5Store
from sevir.constants import ID, FILE_INDEX, FILE_REF, IMG_TYPE


def test_store(store: H5Store) -> None:
    assert len(store.data) == len(store)
    assert {ID, FILE_INDEX, FILE_REF, IMG_TYPE}.issubset(store.catalog.columns)
