import numpy as np
import pandas as pd

from osgeo import gdal
from math import pow


class GeoLocator:

    def __init__(self, src_file):
        self.gt = None
        self.data = None
        self.is_float = False

        self._set_raster_data(src_file)

    def _set_raster_data(self, src_file):
        gd = gdal.Open(src_file)

        self.gt = gd.GetGeoTransform()
        self.data = gd.ReadAsArray().astype(np.float)
        self.is_float = (self.data.dtype in ['float64', 'float32'])

        gd = None

    def get_pixel_val(self, coords):
        result = []
        for coord in coords:
            x = int((coord[0] - self.gt[0])/self.gt[1])
            y = int((coord[1] - self.gt[3])/self.gt[5])

            if self.is_float:
                result.append(round(self.data[y, x], 4))
            else:
                result.append(self.data[y, x])
        return pd.Series(result)


def get_land_index(iterable, ignore=None):
    RS = get_RS(iterable)
    LS = get_LS(iterable)

    return round(LS * RS, 2)

def get_RS(iterable):
    n = len(iterable)
    a = np.array(iterable)
    r = a.prod(dtype=np.int64)/pow(100, n)
    return pow(r, 1/n)

def get_LS(iterable):
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