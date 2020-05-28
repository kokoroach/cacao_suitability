
import os
import pandas as pd
import numpy as np

import cacao_config as CF
import wclim_config as WC
import cacao_raster as _raster
import cacao_plot as _plot

from osgeo import gdal

from cacao_linear_map import LinearMapper
from cacao_land_assessor import get_land_index


def validate(var, period):
    if period == 'baseline':
        pass
    elif period not in WC.PERIODS:
        raise ValueError
    if var not in WC.VARIABLES:
        raise ValueError


def raster_crop(output_file, input_file, no_data=None):
    if not no_data:
        raise Exception
    try:
        gdal.Translate(output_file, input_file,
            format='GTiff', projWin=CF.PH_BOUNDS, noData=no_data, strict=True)
    except Exception as exc:
        os.remove(output_file)


def raster_algebra(output_file, input_files, calc_type=None, no_data=-999):
    if calc_type not in ['sum', 'mean']:
        raise ValueError
    _raster.gdal_calc(output_file, input_files, calc_type, no_data=no_data)


def raster_merge(output_file, input_files, out_format="GTiff", bounds=CF.PH_BOUNDS, a_nodata=CF.NO_DATA_DEFAULT, quiet=False):
    try:
        _raster.gdal_merge(output_file, input_files, out_format=out_format, ul_lr=bounds, a_nodata=a_nodata, quiet=quiet)
    except Exception as exc:
        os.remove(output_file)
        return


def raster_info(input_file, stats=False, as_json=False):
    # NOTE: Disable the creation of aux.xml file
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

    if as_json:
        info = gdal.Info(input_file, stats=stats, deserialize=True)
        return info
    else:
        info = gdal.Info(input_file, stats=stats)
        print(info)


def raster_plot(var, period, raw=False):
    input_file = _get_cropped_file(var, period)
    title, label = _get_plot_details(var, period)
    data_factor = _get_data_factor(var, period)

    _plot.plot(input_file, var=var, title=title, label=label, factor=data_factor, raw=raw)

def raster_delta_plot(var, period):
    input_file = os.path.join(CF.CLIM_VAR_DIR, 'clim_delta', f'delta_{var}_{period}.tif')
    title, label = _get_plot_details(var, period)
    title = f'DELTA {title}'

    # raster_info(input_file, )

    _plot.plot_delta(input_file, var=var, title=title, label=label)


def raster_to_xyz(period):
    lindx_df = pd.DataFrame()

    exc_indexes = exclude(period)
    for var in WC.VARIABLES:
        if period == 'baseline' and var == 'prec':
            src_file = os.path.join(CF.CROPPED_DIR, period, var, f'{var}_sum_{period}.tif')
        else:
            src_file = os.path.join(CF.CROPPED_DIR, period, var, f'{var}_mean_{period}.tif')

        data, no_data = _get_raster_data(src_file)
        for y, x in exc_indexes:
            data[y,x] = no_data

        indexes = np.argwhere(data >= 0)

        # Initialize interpolator object
        mapper = LinearMapper(var)

        data_factor = _get_data_factor(var, period)
        mapped_data = mapper.get_mapped_values(data * data_factor)

        kwargs = {}
        if var == 'prec':
            kwargs['X'] = indexes[:,1]
            kwargs['Y'] = indexes[:,0]

        # Create new column with mapped values
        mapped_values = pd.Series([mapped_data[y,x] for y,x in indexes])
        kwargs[var] = mapped_values.round(2)
        lindx_df = lindx_df.assign(**kwargs)


    # Pre-Process LINDX
    clim_group = lindx_df.filter(items=['prec'])
    temp_group = lindx_df.filter(items=['tmax', 'tmean', 'tmin'])

    # Get Lowest Score in temp group
    temp_group = temp_group.apply(min, axis=1)

    # combine groups
    clim_group = clim_group.assign(temp=temp_group)

    # compute LINDX
    lindx_data = clim_group.apply(get_land_index, axis=1)

    # add LINDX column
    lindx_df = lindx_df.assign(lindx=lindx_data)

    output_dir = os.path.join(CF.LINDX_DIR, period)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_csv = os.path.join(output_dir, f'{period}_lindx.csv')
    lindx_df.to_csv(out_csv, index=False)


# OTHER CORE

def exclude(period):
    # NOTE: Baseline and Future Data do not have the same no. of 
    # NoData values, that is some pixel data are present in one, but
    # not in the other.

    # NOTE: Apply 'per-pixel' intersection to determine 'invalid' pixel

    var = 'prec'
    validate(var, period)

    baseline = os.path.join(CF.CROPPED_DIR, 'baseline', var, f'{var}_sum_baseline.tif')
    future = os.path.join(CF.CROPPED_DIR, '2030s', var, f'{var}_mean_2030s.tif')

    baseline, _ = _get_raster_data(baseline)
    future, _ = _get_raster_data(future)

    if baseline.shape != future.shape:
        raise Exception('Shapes do not match')

    A = (baseline >= 0)
    B = (future >= 0)

    is_baseline = (period == 'baseline')

    data = []

    row, col = A.shape
    for y in range(row):
        for x in range(col):
            if A[y][x] and B[y][x]:
                continue
            elif is_baseline and A[y][x] :
                data.append((y,x))
            elif not is_baseline and B[y][x]:
                data.append((y,x))

    data = np.array(data)

    return data


# PROCESSING

def _get_raster_data(ds_file):
    gd = gdal.Open(ds_file)
    data = gd.GetRasterBand(1).ReadAsArray()
    noData = gd.GetRasterBand(1).GetNoDataValue()
    gd = None
    return data, noData


def _get_calc_type(var):
    if var == 'prec':
        return 'sum'
    elif var in ['tmean', 'tmin', 'tmax']:
        return 'mean'
    else:
        raise ValueError("Unknown variable {}".format(var))


def _get_data_factor(var, period):
    # NOTE: Future Data are INT encoded, i.e. 23.9 is recorded as 239
    if period != 'baseline' and var != 'prec':
        data_factor = 0.1
    else:
        data_factor = 1
    return data_factor


def _get_plot_details(var, period):
    title = '{} '.format(CF.CLIM_TYPES[var])

    if period == 'baseline':
        title += '(Near-Current)'
    elif period == '2030s':
        title += f'(2030)'
    elif period == '2050s':
        title += f'(2050)'

    label = 'Prec (mm)' if var == 'prec' else 'Temp (°C)'

    return title, label


def _get_cropped_file(var, period, model=None):
    if model and model not in WC.MODELS:
        raise ValueError

    validate(var, period)

    calc_type = _get_calc_type(var)
    if period == 'baseline':
        return os.path.join(CF.CROPPED_DIR, period, var, f'{var}_{calc_type}_{period}.tif')
    else:
        if model:
            return os.path.join(CF.CROPPED_DIR, period, var, model, f'{var}_{calc_type}_{period}.tif')
        else:
            return os.path.join(CF.CROPPED_DIR, period, var, f'{var}_mean_{period}.tif')


def _bulk_baseline_cropper(var, noData=-999):
    _var = 'tavg' if var == 'tmean' else var
    for i in range(1, 13):
        src_file = os.path.join(CF.RASTER_DIR, 'baseline', f'wc2.0_30s_{_var}', f'wc2.0_30s_{_var}_{i:02}.tif')        
        out_file = os.path.join(CF.CROPPED_DIR, 'baseline', var, f'{var}_{i}.tif')

        raster_crop(out_file, src_file, noData)


def _bulk_future_merge_cropper(var, period):
    for model in WC.MODELS:
        output_dir = os.path.join(CF.CROPPED_DIR, period, var, model)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(1, 13):
            src_file = os.path.join(CF.RASTER_DIR, period, var, f'{model}_{period}_{var}_^REGION.zip')
            vsi_src = f'/vsizip/{src_file}/{var}_^REGION/{var}_{i}.asc'

            input_files = [
                vsi_src.replace('^REGION', 'b5'),
                vsi_src.replace('^REGION', 'b6') ]

            out_file = os.path.join(output_dir, f'{var}_{i}.tif')

            raster_merge(out_file, input_files, quiet=True)


def _bulk_baseline_algebra(var, cutline=True):
    print(f'PROCESSING: {var}_baseline')

    calc_type = _get_calc_type(var)

    base_dir = os.path.join(CF.CROPPED_DIR, 'baseline', var)
    output_file =  os.path.join(base_dir, f'{var}_{calc_type}_baseline.tif')

    input_files = []
    for i in range(1, 13):
        _file = os.path.join(base_dir, f'{var}_{i}.tif')
        input_files.append(_file)

    path, _ = output_file.rsplit('.', 1)
    temp_file = f'{path}_temp.tif'

    raster_algebra(temp_file, input_files, calc_type=calc_type)

    if cutline:
        gdal.Warp(output_file, temp_file,
                cutlineDSName=CF.SHAPE_FILE, cropToCutline=True, dstAlpha=False)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)


def _bulk_future_algebra_by_model(var, period, model, cutline=True):
    print(f'PROCESSING: {var}_{period}_{model}')

    calc_type = _get_calc_type(var)

    base_dir = os.path.join(CF.CROPPED_DIR, period, var, model)
    output_file = os.path.join(base_dir, f'{var}_{calc_type}.tif')

    input_files = []
    for i in range(1, 13):
        _file = os.path.join(base_dir, f'{var}_{i}.tif')
        input_files.append(_file)

    path, _ = output_file.rsplit('.', 1)
    temp_file = f'{path}_temp.tif'

    try:
        raster_algebra(temp_file, input_files, calc_type=calc_type)
    except Exception:
        os.remove(temp_file)
        return

    if cutline:
        gdal.Warp(output_file, temp_file,
                cutlineDSName=CF.SHAPE_FILE, dstSRS=CF.BASELINE_SRS, cropToCutline=True, dstAlpha=False)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_file)


def _bulk_future_algebra(var, period):
    calc_type = _get_calc_type(var)

    base_dir = os.path.join(CF.CROPPED_DIR, period, var)
    output_file =  os.path.join(base_dir, f'{var}_mean_{period}.tif')

    input_files = []
    for model in WC.MODELS:
        _file = os.path.join(base_dir, model, f'{var}_{calc_type}.tif')
        input_files.append(_file)

    raster_algebra(output_file, input_files, calc_type='mean')


def crop(var, period):
    # 1a. BASELINE: Crop tiff
    # 1b. FUTURE: Merged Crop tiff

    validate(var, period)
    if period == 'baseline':
        _bulk_baseline_cropper(var)
    else:
        _bulk_future_merge_cropper(var, period=period)


def algebra_by_model(var, period, model=None):
    # 2a. FUTURE: Raster Algebra Per Model

    if period == 'baseline':
        raise ValueError('Baseline not applicable')
    if model not in WC.MODELS:
        raise ValueError
    validate(var, period)

    if model:
        _bulk_future_algebra_by_model(var, period, model)
    else:
        for model in WC.MODELS:
            _bulk_future_algebra_by_model(var, period, model)


def algebra(var, period):
    # 2b. BASELINE: Raster Algebra
    # 2b. FUTURE: Raster Algebra Per Model

    validate(var, period)
    if period == 'baseline':
        _bulk_baseline_algebra(var)
    else:
        _bulk_future_algebra(var, period)


def algebraic_delta():
    baseline_excludes = exclude('baseline')
    future_excludes = exclude('2030s')

    for var in WC.VARIABLES:
        if var == 'prec':
            base_dir = os.path.join(CF.CROPPED_DIR, 'baseline', var, f'{var}_sum_baseline.tif')
        else:
            base_dir = os.path.join(CF.CROPPED_DIR, 'baseline', var, f'{var}_mean_baseline.tif')

        base_data, b_notada = _get_raster_data(base_dir)
        for y, x in baseline_excludes:
            base_data[y,x] = b_notada

        indexes = np.argwhere(base_data >= 0)

        for period in WC.PERIODS:
            future_dir = os.path.join(CF.CROPPED_DIR, period, var, f'{var}_mean_{period}.tif')

            future_data, f_nodata = _get_raster_data(future_dir)

            for y, x in future_excludes:
                future_data[y,x] = f_nodata

            for y, x in indexes:
                if var == 'prec':
                    future_data[y,x] = future_data[y,x] - base_data[y,x]
                else:
                    future_data[y,x] = (future_data[y,x] - base_data[y,x]) / 10

            out_path = os.path.join(CF.CLIM_VAR_DIR, 'clim_delta', f'delta_{var}_{period}.tif')

            if var == 'prec':
                no_data = -9999
                future_data = np.where(future_data == f_nodata, no_data, future_data)
                f_nodata = no_data

            if var == 'prec':
                dtype = gdal.GDT_Int32
            else:
                dtype = gdal.GDT_Float32

            _data_to_raster(
                raster_dir=future_dir,
                out_path=out_path,
                data=future_data,
                no_data=f_nodata,
                dtype=dtype
            )


def _data_to_raster(raster_dir=None, out_path=None, data=None, no_data=None, dtype=None):
    """
    Transform data to tiff
    """

    from osgeo import osr

    # PART 1
    gd = gdal.Open(raster_dir)
    gt = gd.GetGeoTransform()
    shape = data.shape
    gd = None

    # Create gtiff
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(out_path, shape[1], shape[0], 1, eType=dtype)

    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(gt)

    # set the reference info
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS(CF.BASELINE_SRS)
    dst_ds.SetProjection(srs.ExportToWkt())

    # write the band
    dst_ds.GetRasterBand(1).SetNoDataValue(no_data)
    dst_ds.GetRasterBand(1).WriteArray(data)



if __name__ == '__main__':
    var = 'tmax'
    period = 'baseline'

    # ----------------------
    # 1. CROPPING
    # ----------------------
    # crop(var, period)


    # ----------------------
    # 2a. RASTER ALGEBRA
    # ----------------------
    # algebra_by_model(var, period)
    # algebra(var, period)

    # ----------------------
    # 3. RASTER ALGEBRA (DIFF)
    # ----------------------
    # algebraic_delta()


    # ----------------------
    # 4. RASTER STATISTICS
    # ----------------------

    # NOTE: Get From Croppped File
    # model = 'bcc_csm1_1'
    # src_file = _get_cropped_file(var, period, model=model)

    # NOTE: Provide specific path
    # src_file = r''
    # raster_info(src_file, stats=True)


    # ----------------------
    # 5. PLOTTING RESULTS
    # ----------------------
    # raster_plot(var, period, raw=True)
    # raster_delta_plot(var='tmax', period='2030s')

    # ----------------------
    # 6. RASTERS TO XYZ (LAND INDEX)
    # ----------------------
    # raster_to_xyz(period)

    # TODO
    # ----------------------
    # 7. LINDX XYZ TO TIFF
    # ----------------------
    # xyz_to_raster(period)


    # ----------------------
    # 8. CREATE MODEL FROM LINDX
    # ----------------------


    # ----------------------
    # 9. LINDX TO TIF
    # ----------------------
    # csv_file =  # path here
    # sample_tif =  # path here
    # lindx_csv_to_tif(csv_file, sample_tif)

