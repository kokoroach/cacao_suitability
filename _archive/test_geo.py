import fiona
import matplotlib.pyplot as plt

from osgeo import gdal
from descartes import PolygonPatch

gdal.UseExceptions()


input_file = r'F:\thesis\cacao_suitability\cropped\baseline\prec\prec_1.tif' 
shape_file = r'F:\thesis\cacao_suitability\ph_shapefile\phl_admbnda_adm3_psa_namria_20180130\phl_admbnda_adm3_psa_namria_20180130.shp'


fig, ax1 = plt.subplots(1,1)

ds = gdal.Open(input_file)
band = ds.GetRasterBand(1)
elevation = band.ReadAsArray()

nrows, ncols = elevation.shape

# I'm making the assumption that the image isn't rotated/skewed/etc. 
# This is not the correct method in general, but let's ignore that for now
# If dxdy or dydx aren't 0, then this will be incorrect
x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()

x1 = x0 + dx * ncols
y1 = y0 + dy * nrows

extent = [x0, x1, y1, y0]

ax1.imshow(elevation, extent=extent)


# FEATURE

with fiona.open(shape_file) as shapefile:
    # name = feature['properties']['ADM1ALT1EN']
    # for feature in shapefile:
        # print(feature["geometry"])
    features = [feature["geometry"] for feature in shapefile]

for feature in features:
    patch = PolygonPatch(feature, edgecolor="black", facecolor="none", linewidth=0.5)
    ax1.add_patch(patch)

plt.show()





# SHAPE HERE


# import fiona
# import rasterio
# import rasterio.plot
# import matplotlib as mpl
# from descartes import PolygonPatch

# input_file = r'F:\thesis\cacao_suitability\cropped\baseline\prec\prec_1.tif' 
# shape_file = r'F:\thesis\cacao_suitability\ph_shapefile\phl_admbnda_adm1_psa_namria_20180130\phl_admbnda_adm1_psa_namria_20180130.shp'


# src = rasterio.open(input_file)
# #
# # for feature in fiona.open(shape_file):
#     # name = feature['properties']['ADM1ALT1EN']
 
# with fiona.open(shape_file) as shapefile:
#     # name = feature['properties']['ADM1ALT1EN']
#     features = [feature["geometry"] for feature in shapefile]


# fig, ax1 = plt.subplots(1,1)

# # import matplotlib.pyplot as plt 
# # from descartes import PolygonPatch

# # BLUE = '#6699cc'
# # fig = plt.figure() 
# # ax = fig.gca() 
# # ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
# # ax.axis('scaled')
# # plt.show()

# ds = gdal.Open(input_file)
# band = ds.GetRasterBand(1)
# elevation = band.ReadAsArray()
# ds = None

# ax1.imshow(elevation)


# plt.show()


