import os
import matplotlib.pyplot as plt
import rasterio as rio

# from rasterio.mask import mask
from rasterio.plot import plotting_extent
from osgeo import gdal

import geopandas as gpd
import earthpy.spatial as es
import earthpy.plot as ep

import wclim_util as util

from config import CMAP, NO_DATA_DEFAULT, SHAPE_FILE, MODULE_DIR, BASELINE_DIR


class WorldClimRasterCropper:

    def __init__(self, ext='tif', clim_type=None, month=None):
        self.clim_types = {
            'prec': 'Precipitation',
            'tmax': 'Max Temperature',
            'tavg': 'Mean Temperature',
            'tmin': 'Min Temperature'
        }
        if clim_type not in self.clim_types:
            raise Exception('Invalid clim type: {}'.format(clim_type))
        if month not in [i for i in range(1, 13)]:
            raise Exception('Invalid month: {}'.format(month))

        self.no_data = 0
        self.no_data_default = NO_DATA_DEFAULT

        self.ext = ext
        self.month = month
        self.clim_type = clim_type

    def _set_negative_color(self, color='w'):
        plt.rcParams['image.cmap'] = CMAP
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_under(color)
        return current_cmap

    def _pre_crop(self, src_tif, shape_file):
        _no_data = 0

        crop_extent = gpd.read_file(shape_file)
        with rio.open(src_tif) as clim_data:
            _no_data = clim_data.nodata
            clim_data, clim_metadata = es.crop_image(clim_data, crop_extent)
        clim_data[0][clim_data[0] == _no_data] = self.no_data_default  # NOTE: set no data

        return clim_data, clim_metadata

    def _plot(self, clim_data_val, clim_data_extent, cmap, title=None, **kwargs):
        # Plot data
        ep.plot_bands(clim_data_val,
            extent=clim_data_extent, cmap=cmap, title=title, scale=False, **kwargs)

    def _crop_extent(self, path_out, clim_data, clim_metadata, clim_data_affine):

        # Update with the new cropped affine info and the new width and height
        clim_metadata.update({
            'transform': clim_data_affine,
            'height': clim_data.shape[1],
            'width': clim_data.shape[2],
            'nodata': self.no_data_default})

        # Write data
        with rio.open(path_out, 'w', **clim_metadata) as ff:
            ff.write(clim_data[0], 1)

        return path_out

    def _warp(self, src_file, dest_file, shape_file):
        ds = gdal.Warp(dest_file,
                       src_file,
                       cutlineDSName=shape_file,
                       cropToCutline=True,
                       dstAlpha=False)
        ds = None

    def crop(self, src_dir, dest_dir, vmin=0, vmax=1000, plot=False):
        status = True

        file_name = f'{self.clim_type}_{self.month:02d}'

        temp_file = os.path.join(dest_dir, f'{file_name}_temp.{self.ext}')
        cropped_file = os.path.join(dest_dir, f'{file_name}.{self.ext}')
        # file_out = os.path.join(dest_dir, f'{file_name}_r.{ext}')

        try:
            # Scans dir and find file_name as substring from dir tree
            src_file = util.get_file_in_dir(file_name, src_dir)

            clim_data, clim_metadata = self._pre_crop(src_file, SHAPE_FILE)
            clim_data_affine = clim_metadata['transform']

            # Create spatial plotting extent for the cropped layer
            clim_data_extent = plotting_extent(clim_data[0], clim_data_affine)

            if plot:
                cmap = self._set_negative_color(color='w')
                title = '{} {}'.format(self.clim_types[clim_type], self.month)

                self._plot(clim_data[0], clim_data_extent, cmap,
                    title=title, vmin=vmin, vmax=vmax)

            self._crop_extent(temp_file, clim_data, clim_metadata, clim_data_affine)
            self._warp(temp_file, cropped_file, SHAPE_FILE)

            os.remove(temp_file)
        
        except Exception as err:
            print(err)
            status = False
        
        return status


if __name__ == "__main__":
    month = 1
    clim_type = 'prec'
    
    src_dir = os.path.join(MODULE_DIR, '_files', 'baseline', 'wc2.0_30s_prec')
    dest_dir = os.path.join(BASELINE_DIR, clim_type)

    cropper = WorldClimRasterCropper(clim_type=clim_type, month=month)
    cropper.crop(src_dir, dest_dir, vmin=0, vmax=1200, plot=True)
