import pandas as pd
import rasterio
import fiona

from config import shape_file

values = pd.Series()

raster = r'F:\thesis\cacao_suit\cropped\baseline\prec\prec_01.tif'

# Read input shapefile with fiona and iterate over each feature
with fiona.open(shape_file) as shp:
    for feature in shp:
        station_name = list(feature['properties'].values())[2]
        _coords = feature['geometry']['coordinates']
        
        coords = []
        for coord in _coords:
            coords.append(coord[0][0])

        # Read pixel value at the given coordinates using Rasterio
        # NB: `sample()` returns an iterable of ndarrays.
        with rasterio.open(raster) as src:
            value = [v for v in src.sample(coords)]

        # Update the pandas serie accordingly
        values.loc[station_name] = value

# Write records into a CSV file
values.to_csv('test.csv', header='Prec', sep=',')


# with rasterio.open(raster) as src:
#     for val in src.sample([(x, y)]): #You can loop through shape coordinates 
#         print(float(val))