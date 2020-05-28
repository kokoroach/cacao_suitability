
import os

# climate variables
CLIM_TYPES = {
    'prec':  'Precipitation',
    'tmax':  'Max Temperature',
    'tmean': 'Mean Temperature',
    'tmin':  'Min Temperature' }

# PH extents in [ulx, uly, lrx, lry]
PH_BOUNDS =  [
    116.92833709716797,
    21.070140838623104,
    126.60534667968749,
    4.586939811706543 ]

BASELINE_SRS = 'WGS84'

# default color
CMAP =  'RdYlGn'

# nodata
NO_DATA_DEFAULT = -999.0

# dirs
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

SHAPE_FILE = os.path.join(MODULE_DIR, 'ph_shapefile', 'country', 'Country.shp')

RASTER_DIR = os.path.join(MODULE_DIR, 'raster_files')
CROPPED_DIR = os.path.join(MODULE_DIR, 'cropped')

MODELS_DIR = os.path.join(MODULE_DIR, 'results', 'models', 'actual_models', 'artifacts')

LINDX_DIR = os.path.join(MODULE_DIR, 'land_indices')

ANN_DIR = os.path.join(MODULE_DIR, 'cacao_ann')

CLIM_VAR_DIR = os.path.join(MODULE_DIR, 'climate_vars')
