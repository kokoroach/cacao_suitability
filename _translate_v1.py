# Import libs
import numpy, os
from osgeo import osr, gdal

# Set file vars
output_file = "out.tif"
# cols = 43200, rows = 18000

# Create gtif
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(output_file, 43200, 18000, 1, gdal.GDT_Byte )
raster = numpy.zeros( (43200, 18000) )

bounds = '126.60833333333335, 21.075000000000003, 116.92500000000001, 4.583333333333336'

# top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
dst_ds.SetGeoTransform( [ 14.97, 0.11, 0, -34.54, 0, 0.11 ] )

# set the reference info 
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS("WGS84")
dst_ds.SetProjection( srs.ExportToWkt() )

# write the band
dst_ds.GetRasterBand(1).WriteArray(raster)