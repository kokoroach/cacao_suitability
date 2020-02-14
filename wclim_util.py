import sys
import os
import inspect

import numpy as np
import rasterio as rio
import earthpy.plot as ep
import matplotlib.pyplot as plt

from osgeo import gdal


# NOTE: Display cropped TIF with arbitrary values in 'coolwarm' color precept
def render(raster_file, title=None, cmap='tab20b', vmin=0, vmax=1, scale=False):
    # img = plt.imread('cropped\prec_chm_cropped.tif')
    # # plt.imshow(img[:, :, 0], cmap=plt.cm.coolwarm)
    # plt.imshow(img[:, :, 0], vmin=0, vmax=800, cmap=plt.cm.coolwarm)
    # plt.colorbar()
    # plt.show()

    plt.rcParams['image.cmap'] = cmap
    current_cmap = plt.cm.get_cmap()
    current_cmap.set_under(color='white')
    # current_cmap.set_under('w', 1)

    with rio.open(raster_file) as rfile:
        data = rfile.read(1)
    
    ep.plot_bands(data,
                cmap=current_cmap,
                title=title,
                vmin=vmin,
                vmax=vmax,
                scale=scale)


def stat(raster_file):
    """
    Get the min, max, mean, stdev based on stats index
    """
    gtif = gdal.Open(raster_file)
    srcband = gtif.GetRasterBand(1)

    # Get raster statistics
    return srcband.GetStatistics(True, True)


def get_value_at_point(raster, pos):
    """
    # Get raster value at given position
    """
    gdata = gdal.Open(raster)
    gt = gdata.GetGeoTransform()
    data = gdata.ReadAsArray().astype(np.float)
    gdata = None

    x = int((pos[0] - gt[0])/gt[1])
    y = int((pos[1] - gt[3])/gt[5])

    return data[y, x]


def get_file_in_dir(self, file_name, src_dir):
    src_file = None

    try:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            for file in filenames:
                if file_name in file:
                    src_file = os.path.join(dirpath, file)
                    break
        if src_file is None:
            raise FileNotFoundError(f'{file_name} not in {src_dir}')
    except Exception as err:
        print(err)

    return src_file


def get_pybase_path():
    for path in sys.path:
        _path = path.split(os.path.sep)
        if _path[-1] == 'Python37-32':
            return path
    raise FileNotFoundError ('Base Path cannot be found')


def get_module_dir():
    _cur_file = inspect.getfile(inspect.currentframe())
    return os.path.dirname(os.path.abspath(_cur_file))


def _makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


if __name__ == "__main__":
    render()
    # stat()
    # raster = r'cropped\baseline\dry\dry_01.tif'
    # pos =  [121.929166666666674,21.0625000000000036]
    # r = get_value_at_point(raster, pos)
    # print(r)


