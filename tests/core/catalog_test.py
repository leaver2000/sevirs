import pytest

from sevirs.constants import IR_069, IR_107, VIS, ImageType
from sevirs.core.catalog import Catalog


@pytest.mark.parametrize(
    "x_types,y_types",
    [
        ((VIS, IR_069), (IR_107,)),
    ],
)
def test_catalog(
    x_types: tuple[ImageType, ...],
    y_types: tuple[ImageType, ...],
) -> None:
    cat = Catalog(img_types=x_types + y_types)
    assert cat.types == x_types + y_types
    x, y = cat.intersect(x_types, y_types)
    assert len(x) + len(y) == len(cat)
    assert cat is not x and cat is not y and x is not y
    assert x.types == x_types
    assert y.types == y_types
    assert cat.id.n_unique() == x.id.n_unique() == y.id.n_unique()
