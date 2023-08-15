import sevirs
from sevirs._lib import GridEncoder
import numpy as np
from scipy.ndimage import shift


def main():
    a = np.random.randint(0, 255, (100, 100)).astype(np.int32)
    e = GridEncoder(a, 5)
    arr = e.to_numpy()
    print(arr[0, 0], a[2, 2])


main()
