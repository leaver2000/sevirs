from sevir.catalog import Catalog
from sevir.constants import VIS, IR_069, IR_107


def test_catalog() -> None:
    cat = Catalog(img_types=[VIS, IR_069, IR_107])
    assert cat.image_set == {VIS, IR_069, IR_107}
    x, y = cat.split_by_types([VIS, IR_069], [IR_107])
    assert len(x) + len(y) == len(cat)
    assert cat is not x and cat is not y and x is not y
    assert x.image_set == {VIS, IR_069}
    assert y.image_set == {IR_107}
    vis = cat.get_by_img_type([VIS])
    assert cat is not vis and cat.image_set == {VIS, IR_069, IR_107}
    vis = cat.get_by_img_type([VIS], inplace=True)
    assert cat is vis and cat.image_set == {VIS}
