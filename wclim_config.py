
import os

from wclim_util import get_pybase_path, get_module_dir, _makedirs


# PH extents in [ulx, uly, lrx, lry]
ph_bounds =  [
    116.92500000000001,
    21.075000000000003,
    126.60833333333335,
    4.583333333333336
]

cmap =  'RdYlGn'

no_data_default = -999.0

module_dir = get_module_dir()
pybase_path = get_pybase_path()

shape_file = os.path.join(module_dir, 'ph_shapefile', 'Country.shp')

script_dir = os.path.join(pybase_path, 'Scripts')

baseline_dir = os.path.join(module_dir, 'cropped', 'baseline')
ft_2030_dir = os.path.join(module_dir, 'cropped', '2030')
ft_2050_dir = os.path.join(module_dir, 'cropped', '2050')
