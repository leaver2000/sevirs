import numpy as np
import polars as pl
from sevirs.data.generic import DataGenerator
class MyGenerator(DataGenerator[str, np.ndarray]):
    def __init__(self):
        self.data = data = {
            "a": np.array([1, 2, 3]),
        }
        
        super().__init__(data.keys())

    def get(self, key) -> np.ndarray:
        return self.data[key]
        
    def get_metadata(self, index: str | None = None) -> pl.DataFrame:
        return pl.from_dict(self.data)
        
def test_datagenerator() -> None:
    gen = MyGenerator()
    assert gen.get("a").tolist() == [1, 2, 3]
    assert gen.get_metadata().shape == (3, 1)
    for value in gen:
        assert isinstance(value, np.ndarray)
        # assert value in gen.data
        # assert key in gen.get_metadata().columns
