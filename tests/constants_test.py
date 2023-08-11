import pytest

from sevirs.constants import IR_069, IR_107, LGHT, VIL, VIS, ImageType


@pytest.mark.parametrize(
    "img_type,name",
    [
        (VIS, "vis"),
        (VIL, "vil"),
        (IR_069, "ir069"),
        (IR_107, "ir107"),
        (LGHT, "lght"),
    ],
)
def test_image_type(img_type: ImageType, name: str) -> None:
    assert name == img_type == ImageType(name) == ImageType(img_type.name)  # type: ignore
