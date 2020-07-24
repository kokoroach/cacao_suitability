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

import sys

from osgeo import gdal

try:
    import numpy as Numeric
except ImportError:
    import Numeric

# =============================================================================

# NOTE: Program modified from gdal2xyz.py
def gdal2xyz(srcfile, dstfile=None, clipped=False, geoformat=False, apply_factor=None, mapper=None):
    """
    params:
    clipped : bool : `True` means to exclude noData values in the output
    geoformat: bool : enable geoloc formatting
    mapper : function : set the mapper function for values. `None` evaluates to actual value
    """

    srcwin = None
    skip = 1
    band_nums = []
    delim = ','

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
    
    # NOTE: Custom band format
    if apply_factor != 1:
        band_format = (("%g" + delim) * len(bands)).rstrip(delim) + '\n'

    # Setup an appropriate print format.
    if abs(gt[0]) < 180 and abs(gt[3]) < 180 \
       and abs(srcds.RasterXSize * gt[1]) < 180 \
       and abs(srcds.RasterYSize * gt[5]) < 180:
        frmt = '%.10g' + delim + '%.10g' + delim + '%s'
    else:
        frmt = '%.3f' + delim + '%.3f' + delim + '%s'

    # NOTE: Custom changes here

    # noData 
    noDataValue = srcds.GetRasterBand(1).GetNoDataValue()
    
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

            # NOTE: Exclude
            if clipped and noDataValue in x_i_data:
                continue

            # NOTE: Linear Mapping Values
            if mapper:
                mapped_data = []
                for i_data in x_i_data:
                    if apply_factor != 1:
                        i_data *= apply_factor
                    mapped_data.append(mapper(i_data))
                band_str = band_format % tuple(mapped_data)
            else:
                if apply_factor != 1:
                    x_i_data = [i* apply_factor for i in x_i_data]
                band_str = band_format % tuple(x_i_data)
            if geoformat:
                line = frmt % (float(geo_x), float(geo_y), band_str)
            else:
                frmt = '%s' + delim + '%s' + delim + '%s'
                line = frmt % (geo_x, geo_y, band_str)

            dst_fh.write(line)
