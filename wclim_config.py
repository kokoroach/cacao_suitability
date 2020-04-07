
import os

from wclim_util import get_pybase_path, get_module_dir, _makedirs


CLIM_TYPES = ['prec', 'tmax', 'tavg', 'tmin', 'dry']

# PH extents in [ulx, uly, lrx, lry]
PH_BOUNDS =  [
    116.92500000000001,
    21.075000000000003,
    126.60833333333335,
    4.583333333333336
]

CMAP =  'RdYlGn'

NO_DATA_DEFAULT = -999.0

MODULE_DIR = get_module_dir()
PYBASE_PATH = get_pybase_path()

SHAPE_FILE = os.path.join(MODULE_DIR, 'ph_shapefile', 'Country.shp')

SCRIPT_DIR = os.path.join(MODULE_DIR, 'Scripts')

BASELINE_DIR = os.path.join(MODULE_DIR, 'cropped', 'baseline')
FT_2030_DIR = os.path.join(MODULE_DIR, 'cropped', '2030')
FT_2050_DIR = os.path.join(MODULE_DIR, 'cropped', '2050')
