from pathlib import Path

# # PH extents in [ulx, uly, lrx, lry]
PH_BOUNDS = [
    116.92833709716797,
    21.070140838623104,
    126.60534667968749,
    4.586939811706543
]

BASELINE_SRS = 'WGS84'

# default color scheme
CMAP = 'RdYlGn'

# nodata
NO_DATA_DEFAULT = -999

# Base dirs
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent

# Shapefiles
SHAPEFILES_DIR = PROJECT_DIR / 'shapefiles'
PH_BORDER = SHAPEFILES_DIR / 'country' / 'Country.shp'
PROVINCIAL_BORDER = SHAPEFILES_DIR / 'provinces' / 'Provinces.shp'

# Set data directories
DATA_DIR = PROJECT_DIR / 'data'
CLIMATE_DIR = DATA_DIR / 'climate'

# Set Climate data I/O
INPUT_CLIM_DIR = CLIMATE_DIR / 'input'
OUTPUT_CLIM_DIR = CLIMATE_DIR / 'output'

# Baseline and Future Data
CCFAS_DATA_DIR = INPUT_CLIM_DIR / 'ccfas'
WCLIM_DATA_DIR = INPUT_CLIM_DIR / 'wclim'

# Preprocessed Data
CROPPED_DIR = INPUT_CLIM_DIR / 'cropped'

# Output Dirs
BASELINE_OUT_DIR = OUTPUT_CLIM_DIR / 'baseline'
FUTURE_OUT_DIR = OUTPUT_CLIM_DIR
