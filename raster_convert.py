#!/usr/bin/env python
###############################################################################
# $Id$
#
# Project:  GDAL
# Purpose:  Script to translate GDAL supported raster into XYZ ASCII
#           point stream.
# Author:   Frank Warmerdam, warmerdam@pobox.com
#
###############################################################################
# Copyright (c) 2002, Frank Warmerdam <warmerdam@pobox.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
###############################################################################

# NOTE: Program modified from gdal2xyz.py


import sys
import os

from osgeo import gdal
from wclim_config import baseline_dir

try:
    import numpy as Numeric
except ImportError:
    import Numeric


def gdal2xyz(srcfile, dstfile=None, clim_type=None, noData=-999, exclude=False, delim=','):

    srcwin = None
    skip = 1
    band_nums = []

    gdal.AllRegister()

    if clim_type is None or clim_type not in ['prec', 'tmax', 'tavg', 'tmin', 'dry']:
        print('Unknown clim_type: %s.' % clim_type)
        sys.exit(1)

    if band_nums == []:
        band_nums = [1]

    # Open source file.
    srcds = gdal.Open(srcfile)
    if srcds is None:
        print('Could not open %s.' % srcfile)
        sys.exit(1)

    bands = []
    for band_num in band_nums:
        band = srcds.GetRasterBand(band_num)
        if band is None:
            print('Could not get band %d' % band_num)
            sys.exit(1)
        bands.append(band)

    gt = srcds.GetGeoTransform()

    # Collect information on all the source files.
    if srcwin is None:
        srcwin = (0, 0, srcds.RasterXSize, srcds.RasterYSize)

    # Open the output file.
    if dstfile is not None:
        dst_fh = open(dstfile, 'wt')
    else:
        dst_fh = sys.stdout

    dt = srcds.GetRasterBand(1).DataType
    if dt == gdal.GDT_Int32 or dt == gdal.GDT_UInt32:
        band_format = (("%d" + delim) * len(bands)).rstrip(delim) + '\n'
    else:
        band_format = (("%g" + delim) * len(bands)).rstrip(delim) + '\n'

    # Setup an appropriate print format.
    if abs(gt[0]) < 180 and abs(gt[3]) < 180 \
       and abs(srcds.RasterXSize * gt[1]) < 180 \
       and abs(srcds.RasterYSize * gt[5]) < 180:
        frmt = '%.10g' + delim + '%.10g' + delim + '%s'
    else:
        frmt = '%.3f' + delim + '%.3f' + delim + '%s'

    # Loop emitting data.
    _i = 0
    for y in range(srcwin[1], srcwin[1] + srcwin[3], skip):

        data = []
        for band in bands:

            band_data = band.ReadAsArray(srcwin[0], y, srcwin[2], 1)
            band_data = Numeric.reshape(band_data, (srcwin[2],))
            data.append(band_data)

        for x_i in range(0, srcwin[2], skip):

            x = x_i + srcwin[0]

            geo_x = gt[0] + (x + 0.5) * gt[1] + (y + 0.5) * gt[2]
            geo_y = gt[3] + (x + 0.5) * gt[4] + (y + 0.5) * gt[5]

            x_i_data = []
            for i in range(len(bands)):
                x_i_data.append(data[i][x_i])

            # Exclude
            if exclude and noData in x_i_data:
                continue
            _i += 1
    print(_i)
            # band_str = band_format % tuple(x_i_data)

            # line = frmt % (float(geo_x), float(geo_y), band_str)
        
            # dst_fh.write(line)



if __name__ == '__main__':
    clim_type = 'dry'

    for i in range(1, 13):
        src_file = os.path.join(baseline_dir, 'tif', clim_type, '{}_{:02d}.tif'.format(clim_type, i))
        dest_file =  os.path.join(baseline_dir, 'csv', clim_type, '{}_{:02d}.csv'.format(clim_type, i))
        
        # gdal2xyz(src_file, dstfile=dest_file, clim_type=clim_type, exclude=True)
        gdal2xyz(src_file, clim_type=clim_type, exclude=True)
