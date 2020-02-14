import pandas as pd
import rasterio
import fiona

values = pd.Series()

# Read input shapefile with fiona and iterate over each feature
with fiona.open('ph_shapefile\Country.shp') as shp:
    
    for feature in shp:
        station_name = list(feature['properties'].values())[2]
        coords = feature['geometry']['coordinates']

        # Read pixel value at the given coordinates using Rasterio
        # NB: `sample()` returns an iterable of ndarrays.
        # print(len(coords))
        for item in enumerate(coords):
            print('-->')
            print(item[0])
            if i == 10:
                break
            if len(item) != 1:
                print('here')
        
        with rasterio.open('prec_nimr_gcm_v2.tif') as src:
            for v, x  in src.sample(coords):
                value = [v for v in src.sample([coords])][0][0]

        # # Update the pandas serie accordingly
        values.loc[station_name] = value

# Write records into a CSV file
values.to_csv('test.csv')