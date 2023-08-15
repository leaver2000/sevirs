# flake8: noqa
from ._typing import Array, Nd, N
import numpy as np

class GridEncoder:
    def __init__(self, data: Array[Nd[N, N], np.int32], n: int) -> None: ...
    def to_numpy(self) -> Array[Nd[N, N, N, N], np.int32]: ...
