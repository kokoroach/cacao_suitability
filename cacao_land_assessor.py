import numpy as np
import pandas as pd

from osgeo import gdal
from math import pow


def get_land_index(iterable, ignore=None):
    RS = _get_RS(iterable)
    LS = _get_LS(iterable)

    return round(LS * RS, 2)

def _get_RS(iterable):
    n = len(iterable)
    a = np.array(iterable)
    r = a.prod(dtype=np.int64)/pow(100, n)
    return pow(r, 1/n)

def _get_LS(iterable):
    n = len(iterable)
    result = 0
    for i, j in enumerate(iterable):
        if i == 0:
            result = pow(j, 1/n)
        else:
            result *= pow(j, 1/n)
    return result


if __name__ == '__main__':
    data = [80.62, 86.0]
    r = get_land_index(data)
    print(r)