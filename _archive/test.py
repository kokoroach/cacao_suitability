

# S1 =    [68961,    19.67,    32978,      9.41,     24619,      7.02]
# S2  =   [239877,   68.43,    269025,    76.75,    274913,      78.43]
# S3  =   [29487,    8.41,     37862,      10.8,     41254,      11.77]
# N2  =   [5666,     1.62,     5404,       1.54,      5284,       1.51]
# N1   =  [6552,     1.87,     5274,        1.5,      4473,       1.28]


# for A in [S1, S2, S3, N2, N1]:
#     u_2030 = A[2] - A[0]
#     p_2030 = round(A[3] - A[1], 2)

#     u_2050 = A[4] - A[0]
#     p_2050 = round(A[5] - A[1], 2)

#     print(u_2030, p_2030, u_2050, p_2050)

import numpy as np  
import matplotlib.pyplot as plt

from osgeo import gdal


def plot():
    input_file = r'F:\thesis\cacao_suitability\cropped\baseline\prec\prec_1.tif' 

    fig, ax1 = plt.subplots(1,1)

    ds = gdal.Open(input_file)
    ds_arr = np.array(ds.GetRasterBand(1).ReadAsArray()) 

    # im = ax1.imshow(ds_arr)
    # cb = fig.colorbar(im, ax=ax1)


    # ------------------------------
    nrows, ncols = ds_arr.shape

    # I'm making the assumption that the image isn't rotated/skewed/etc. 
    # This is not the correct method in general, but let's ignore that for now
    # If dxdy or dydx aren't 0, then this will be incorrect
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()

    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows
    # ------------------------------

    # import matplotlib.pyplot as plt

    # grid=[x for x in range(10)]
    # graphs=[
    #     [500,500,500,500,500,500,500,500,500,500],
    #     # [1,1,1,5,5,5,3,5,6,0],
    #     # [1,1,1,0,0,3,3,2,4,0],
    #     # [1,2,4,4,3,2,3,2,4,0],
    #     # [1,2,3,3,4,4,3,2,6,0],
    #     # [1,1,3,3,0,3,3,5,4,3],
    # ]

    # for gg,graph in enumerate(graphs):
    #     plt.plot(grid, graph, label='g'+str(gg))
    # plt.legend(loc=3,bbox_to_anchor=(1,0))
    # # plt.show()
    extent = [x0, x1, y1, y0]
    print(extent)
    im = plt.imshow(ds_arr, cmap='gist_earth', extent=extent)
    cb = fig.colorbar(im, ax=ax1)


    plt.tight_layout()
    plt.show()

def overlay():
    import fiona

    shapefile = r'F:\thesis\cacao_suitability\ph_shapefile\Country.shp'
    shape = fiona.open(shapefile, 'r')
    # print(dir(shape))
    for item in shape.items():
        # print(item[1].keys())
        print(item[1]['type'])
        print(item[1]['id'])
        print(item[1]['properties'])
        # print(item[1].keys())
        # print(len(item[1]))
        # print(len(item))
        # print(type(item))
        # print(dir(item))
        exit()
    # res = shape.bounds   # (minX, minY, maxX, maxY)


if __name__ == '__main__':
    # plot()
    # overlay()

    # Make a figure twice as tall as it is wide::
    # from matplotlib.figure import Figure, figaspect

    # w, h = figaspect(2.)
    # fig = Figure(figsize=(w, h))
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(A,)

    # # Make a figure with the proper aspect for an array::
    # from random import rand

    # A = rand(5, 3)
    # w, h = figaspect(A)
    # fig = Figure(figsize=(w, h))
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.imshow(A,)

    from osgeo import gdal
    gdal.UseExceptions()

    input_file = r'F:\thesis\cacao_suitability\cropped\baseline\prec\prec_1.tif' 

    ds = gdal.Open(input_file)
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    nrows, ncols = elevation.shape

    # I'm making the assumption that the image isn't rotated/skewed/etc. 
    # This is not the correct method in general, but let's ignore that for now
    # If dxdy or dydx aren't 0, then this will be incorrect
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    print(ds.GetGeoTransform())

    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows

    print(x1)
    print(y1)

    # [longitude_top_left,longitude_top_right,latitude_bottom_left,latitude_top_left]
    # print([x0, x1, y0, y1])
    # ax.set_xlim((coord["minLong"], coord["maxLong"]))
    # ax.set_ylim((coord["minLat"], coord["maxLat"]))

    fig, ax = plt.subplots()
    ax.set_xlim(y0, y1)
    ax.set_ylim(x0, x1)

    plt.imshow(elevation)
    plt.show()

