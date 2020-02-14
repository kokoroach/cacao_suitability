import os
from osgeo import gdal

# file_path = r'cropped_future\nimr_hadgem2\prec_1'

def transform(file_path):
    inDs = gdal.Open('{}.tif'.format(file_path))
    outDs = gdal.Translate('{}.xyz'.format(file_path), inDs, noData=-999, format='XYZ', creationOptions=["ADD_HEADER_LINE=YES"])
    outDs = None
    try:
        os.remove('{}.csv'.format(file_path))
    except OSError:
        pass
    os.rename('{}.xyz'.format(file_path), '{}.csv'.format(file_path))

    # os.system('ogr2ogr -f "ESRI Shapefile" -oo X_POSSIBLE_NAMES=X* -oo Y_POSSIBLE_NAMES=Y* -oo KEEP_GEOM_COLUMNS=NO {0}.shp {0}.csv'.format(filename))
    # commands = '-f CSV -sql "SELECT x, y, z FROM output_XYZ_with_NoData WHERE z != \'-32768\'" output_XYZ_without_NoData.csv {}.csv -lco SEPARATOR=SPACE'.format(file_path)
    # cmd = ['-f', 'CSV' '-sql' "SELECT x, y, z FROM output_XYZ_with_NoData WHERE z != '-32768'", 'output_XYZ_without_NoData.csv', '{}.csv'.format(file_path), '-lco' 'SEPARATOR=SPACE']
    # os.system('ogr2ogr {}'.format(commands))


    # gdal_translate -of "XYZ" {} {} -a_nodata "value [-9999]"

if __name__ == "__main__":
    for i in range(1, 13):
        clim_type = 'prec'
        file_path = f'cropped\\baseline\\{clim_type}\\{clim_type}_{i:02d}_count'
        print(file_path)

        transform(file_path)
        exit()

    