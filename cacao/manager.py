import os
import warnings

from osgeo import gdal
from pprint import pprint

from cacao import config as CF
from cacao import raster
from climate.ccfas.config import Model, Region, Period, Climate
from climate.wclim.downloader import get_output_file as get_wclim_file


class utils:

    @staticmethod
    def get_calc_type(clim):
        match clim:
            case Climate.PREC.value:
                return 'sum'
            case Climate.TMAX.value | Climate.TMEAN.value | Climate.TMIN.value:
                return 'mean'
            case _:
                raise ValueError("Unknown climate type: {}".format(clim))

    @staticmethod
    def get_croppped_data_dir(clim, period, model=None):
        output_dir = None

        match period:
            case Period.P_BASELINE.value:
                output_dir = CF.CROPPED_DIR / period / clim

            case Period.P_2030.value | Period.P_2050.value:
                output_dir = CF.CROPPED_DIR / period / clim / model

            case _:
                raise ValueError('Invalid period passed. Was %s' % period)

        os.makedirs(output_dir, exist_ok=True)
        return output_dir


# ----------------------
# CROPPING
# ----------------------
def generate_raster_by_merge(clim, period, model=None):
    if model:
        merge_rasters(clim=clim, period=period, model=model)
    else:
        # If model is not provided, download all models
        for _model in Model.values():
            merge_rasters(clim=clim, period=period, model=_model)


def merge_rasters(clim=None, period=None, model=None):
    """
    Merge the raster files from Regions' b5 and b6 to get the PH data
    """
    print('Generating future rasters for: {}:{}:{}'.format(clim, period, model))

    output_dir = utils.get_croppped_data_dir(clim, period, model)

    for i in range(1, 13):
        src_file = os.path.join(CF.CCFAS_DATA_DIR, period, clim, f'{model}_{period}_{clim}_^REGION.zip')  # noqa
        vsi_src = f'/vsizip/{src_file}/{clim}_^REGION/{clim}_{i}.asc'
        output_file = output_dir / f'{clim}_{i}.tif'

        if os.path.exists(output_file):
            message = "Skipping existing file: %s" % output_file.name
            warnings.warn(message)
            continue

        input_files = [vsi_src.replace('^REGION', Region.B5.value),
                       vsi_src.replace('^REGION', Region.B6.value)]

        with open(output_file, 'w+'):
            pass

        try:
            _merge_rasters(output_file, input_files, verbose=True)
        except Exception as exc:
            os.remove(output_file)
            raise exc


def _merge_rasters(output_file, input_files, out_format=None, bounds=None,
                   a_nodata=None, verbose=True):

    out_format = out_format or "GTiff"
    bounds = bounds or CF.PH_BOUNDS
    a_nodata = a_nodata or CF.NO_DATA_DEFAULT

    raster.merge(output_file, input_files, out_format=out_format,
                 ul_lr=bounds, a_nodata=a_nodata, verbose=verbose)


def generate_raster_by_crop(clim, nodata=None):
    print('Generating baseline rasters for: {}'.format(clim))

    _clim = 'tavg' if clim == Climate.TMEAN.value else clim

    output_dir = utils.get_croppped_data_dir(clim, Period.P_BASELINE.value)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 13):
        src_file = get_wclim_file(clim)
        src_file = f'/vsizip/{src_file}/wc2.1_30s_{_clim}_{i:02}.tif'

        output_file = output_dir / f'{clim}_{i}.tif'
        with open(output_file, 'w+'):
            pass

        src_file = str(src_file)
        output_file = str(output_file)

        try:
            raster_crop(src_file, output_file, nodata)
        except Exception:
            os.remove(output_file)
            raise


def raster_crop(input_file, output_file, noData=None):
    noData = noData or CF.NO_DATA_DEFAULT

    gdal.Translate(output_file, input_file, format='GTiff',
                   projWin=CF.PH_BOUNDS, noData=noData, strict=True)


# ----------------------
# RASTER ALGEBRA CALCULATIONS
# ----------------------

def do_baseline_raster_calc(clim):
    print(f'Processing raster calculation for: {clim}:baseline')

    calc_type = utils.get_calc_type(clim)

    input_dir = utils.get_croppped_data_dir(clim, Period.P_BASELINE.value)
    output_file = CF.BASELINE_OUT_DIR / f'{clim}_{calc_type}.tif'

    os.makedirs(output_file.parent, exist_ok=True)

    input_files = []
    for i in range(1, 13):
        clim_file = input_dir / f'{clim}_{i}.tif'
        input_files.append(clim_file)

    output_file = str(output_file)
    path, _ = output_file.rsplit('.', 1)
    temp_file = f'{path}_temp.tif'

    try:
        raster.gdal_calc(temp_file, input_files, calc_type, no_data=-999)
    except Exception:
        os.remove(temp_file)
        return

    # Crop the final result
    crop_ph_only(output_file, temp_file)


def do_future_raster_calc(clim, period, model=None):
    if model:
        _future_raster_calc(clim=clim, period=period, model=model)
    else:
        # If model is not provided, download all models
        for _model in Model.values():
            _future_raster_calc(clim=clim, period=period, model=_model)


def _future_raster_calc(clim, period, model):
    print(f'Processing raster calculation for: {clim}:{period}:{model}')

    calc_type = utils.get_calc_type(clim)

    input_dir = utils.get_croppped_data_dir(clim, period, model=model)
    output_file = CF.FUTURE_OUT_DIR / period / clim / f'{clim}_{calc_type}_{model}.tif'  # noqa

    # Do not process existing file
    if os.path.exists(output_file):
        message = "Skipping existing file: %s" % output_file.name
        warnings.warn(message)
        return

    os.makedirs(output_file.parent, exist_ok=True)

    input_files = []
    for i in range(1, 13):
        clim_file = input_dir / f'{clim}_{i}.tif'
        input_files.append(clim_file)

    output_file = str(output_file)
    path, _ = output_file.rsplit('.', 1)
    temp_file = f'{path}_temp.tif'

    try:
        raster.gdal_calc(temp_file, input_files, calc_type=calc_type)
    except Exception:
        os.remove(temp_file)
        return

    crop_ph_only(output_file, temp_file,
                 srcSRS=CF.BASELINE_SRS, dstSRS=CF.BASELINE_SRS)


def generate_future_calc_summary(clim, period):
    input_files = []

    calc_type = utils.get_calc_type(clim)
    src_dir = CF.FUTURE_OUT_DIR / period / clim

    # Validate that expected models exists:
    for model in Model.values():
        file_path = src_dir / f'{clim}_{calc_type}_{model}.tif'
        input_files.append(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                'Unable to find expected file: %s' % file_path)

    output_file = CF.FUTURE_OUT_DIR / period / f'{clim}_{calc_type}.tif'

    # Do not process existing file
    if os.path.exists(output_file):
        message = "Skipping existing file: %s" % output_file.name
        warnings.warn(message)
        return

    os.makedirs(output_file.parent, exist_ok=True)

    output_file = str(output_file)
    raster.gdal_calc(output_file, input_files, calc_type='mean')


def crop_ph_only(output_file, input_file, srcSRS=None, dstSRS=None):
    """
    Crop GeoTiff file to only include the PH Territory
    """
    # print('CROPPING: ', output_file, input_file, dst_srs)
    gdal.Warp(output_file, input_file, cutlineDSName=CF.PH_BORDER,
              cropToCutline=True, dstAlpha=False, srcSRS=srcSRS, dstSRS=dstSRS)
    os.remove(input_file)


# ----------------------------
# RASTER STATISTICS
# ----------------------------

def raster_info(clim, period, src_file=None):
    """
    Primarily used for getting info for preprocessed rasters. That is, the args
    for clim, period will check data in. the CF.OUTPUT_DATA_DIR.

    To override for specific file, pass `scrc_file` path instead.
    """
    calc_type = utils.get_calc_type(clim)

    if src_file is None:
        file_name = f'{clim}_{calc_type}.tif'
        src_file = CF.OUTPUT_DATA_DIR / period / file_name
        src_file = str(src_file)

    # NOTE: Disable the creation of aux.xml file
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

    info = gdal.Info(src_file, stats=True, format='json')
    pprint(info, compact=True)
