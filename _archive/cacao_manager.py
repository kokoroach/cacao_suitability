
import os
import pandas as pd
import numpy as np

import cacao_raster as _raster
import cacao_plot as _plot
import cacao_config as CF
import wclim_config as WC

from osgeo import gdal


def _get_apply_factor(factor):
    if factor == 0:
        raise
    if factor == None:
        return 1
    return factor 


def raster_crop(output_file, input_file, no_data=-999):
    try:
        gdal.Translate(output_file, input_file,
            format='GTiff', projWin=CF.PH_BOUNDS, noData=no_data, strict=True)
    except Exception as exc:
        os.remove(output_file)
        return


def raster_algebra(output_file, input_files, calc_type=None, no_data=-999):
    if calc_type not in ['sum', 'mean']:
        raise
    _raster.gdal_calc(output_file, input_files, calc_type, no_data=no_data)


def raster_plot(input_file, var=None, title=None, label=None, linx=False, apply_factor=None):
    apply_factor = _get_apply_factor(apply_factor)
    _plot.plot(input_file, var=var, title=title, label=label, linx=linx, apply_factor=apply_factor)


def raster_merge(output_file, input_files, out_format="GTiff", bounds=CF.PH_BOUNDS, a_nodata=CF.NO_DATA_DEFAULT, quiet=False):
    try:
        _raster.gdal_merge(output_file, input_files, out_format=out_format, ul_lr=bounds, a_nodata=a_nodata, quiet=quiet)
    except Exception as exc:
        os.remove(output_file)
        return


def raster_compare(base_file, new_file):
    _raster.gdal_compare(base_file, new_file)


def raster_to_xyz(src_file, output_file=None, clipped=False, geoformat=False, apply_factor=None, mapper=None):
    from cacao_gdal2xyz import gdal2xyz
    apply_factor = _get_apply_factor(apply_factor)
    gdal2xyz(src_file, dstfile=output_file, clipped=clipped, geoformat=geoformat, apply_factor=apply_factor, mapper=mapper)


def raster_info(input_file, stats=False):
    info = gdal.Info(input_file, stats=stats)
    print(info)


def shape_extent(shapefile, package='fiona'):
    # return : list :  [minX, minY, maxX, maxY]

    if package == 'fiona':
        import fiona
        shape = fiona.open(shapefile, 'r')
        res = shape.bounds   # (minX, minY, maxX, maxY)
        return list(res)

    elif package == 'ogr':
        from osgeo import ogr
        file = ogr.Open(shapefile)
        shape = file.GetLayer(0)
        feature = shape.GetFeature(0)
        geom = feature.GetGeometryRef()
        (minX, maxX, minY, maxY) = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
        res = [minX, minY, maxX, maxY]
        return res

    elif package == 'geopd':
        import geopandas as gpd
        shapefile = gpd.read_file(shapefile)
        res = shapefile.total_bounds  # numpy.ndarray[minX minY maxX maxY]
        return res.tolist()


def lindx_csv_to_tif(csv_file, sample_tif):
    # Import libs
    from osgeo import osr

    def _set_pixel_val(gt, data, XYZ):
        for i, (c_x, x_y, c_z) in XYZ.iterrows():
            x = int((c_x - gt[0])/gt[1])
            y = int((x_y - gt[3])/gt[5])

            data[y, x] = c_z

    # PART 1
    gd = gdal.Open(sample_tif)
    gt = gd.GetGeoTransform()
    shape = gd.ReadAsArray().shape
    data = np.full(shape, -999., dtype='float32')
    gd = None

    # PART 2
    df = pd.read_csv(csv_file, header=0, usecols=['X','Y','lindx'])
    df = df.rename(columns={"lindx":"Z"})

    _set_pixel_val(gt, data, df)

    # Set file vars
    head, _ = os.path.splitext(csv_file)
    output_file = f'{head}.tif'

    # Create gtif
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_file, shape[1], shape[0], 1, eType=gdal.GDT_Float32)

    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(gt)

    # set the reference info 
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS(CF.BASELINE_SRS)
    dst_ds.SetProjection(srs.ExportToWkt())

    # write the band
    # TODO Set noData
    dst_ds.GetRasterBand(1).SetNoDataValue(-999)
    dst_ds.GetRasterBand(1).WriteArray(data)


def rasters_to_lidx_xyz(period):
    if period in ['baseline', '2030s', '2050s']:
        base_csv = os.path.join(CF.CROPPED_DIR, period, 'tmin', f'tmin_mean_{period}.csv')
    else:
        raise ValueError(f"Unknown Period: {period}")

    if not os.path.exists(base_csv):
        raise FileNotFoundError(f"Create base csv on {base_csv}")

    from cacao_land_index import GeoLocator, get_land_index
    from cacao_linear_map import LinearMapper


    lindx_df = pd.read_csv(base_csv,
        header=None, usecols=[0,1], names=['X', 'Y'])

    # Add var columns
    for var in CF.CLIM_TYPES:
        if period == 'baseline' and var == 'prec':
            src_file = os.path.join(CF.CROPPED_DIR, period, var, f"{var}_sum_{period}.tif")
        else:
            src_file = os.path.join(CF.CROPPED_DIR, period, var, f"{var}_mean_{period}.tif")

        # Initialize necessary objects
        locator = GeoLocator(src_file) 
        mapper = LinearMapper(var)

        pix_vals = locator.get_pixel_val(zip(lindx_df['X'], lindx_df['Y']))
        # NOTE: Assertion of apply factor
        if period != 'baseline' and var != 'prec':
            pix_vals *= 0.1
        mapped_vals = mapper.get_mapped_values(pix_vals)

        # Create new column with mapped values
        kwargs = {var: mapped_vals}
        lindx_df = lindx_df.assign(**kwargs)

    # Pre-Process LINDX
    clim_group = lindx_df.filter(items=['prec'])
    temp_group = lindx_df.filter(items=['tmax', 'tmean', 'tmin'])

    # Get Lowest Score in temp group
    temp_group = temp_group.apply(min, axis=1)

    # combine groups 
    clim_group = clim_group.assign(temp=temp_group)

    # compute lindx
    lindx_data = clim_group.apply(get_land_index, axis=1)

    # Add LINDX column
    lindx_df = lindx_df.assign(lindx=lindx_data)

    # TO CSV
    out_csv = os.path.join(CF.CROPPED_DIR, period, f'{period}_lindx.csv')
    lindx_df.to_csv(out_csv, index=False)


def get_lindx_tif_stat(tif_file):
    from raster_util.intersections import intersect

    bounds = [0, 12.5, 25, 50, 75, 100]
    n = len(bounds)-1

    gd = gdal.Open(tif_file)
    gt = gd.GetGeoTransform()
    noData = gd.GetRasterBand(1).GetNoDataValue()
    indexes = gd.GetRasterBand(1).ReadAsArray()
    gd = None


    baseline, future = intersect()
    is_baseline = ('baseline' in tif_file)
    exclusion = baseline if is_baseline else future

    for point in exclusion:
        x, y = point
        indexes[y][x] = noData
    
    indx_len = len(indexes[indexes >= 0])

    stats = []
    for i in range(n):
        lbd, rbd = bounds[i], bounds[i+1]
        if i == n:
            lmask = indexes >= rbd
            rmask = indexes <= lbd
            bnd = f"[{lbd}, {rbd}]"
        else:
            lmask = indexes >= bounds[i]
            rmask = indexes < bounds[i+1]
            bnd = f"[{lbd}, {rbd})"
        
        counts = (lmask & rmask).sum()
        
        out = {}
        out['bounds'] = bnd
        out['counts'] = counts
        out['%'] = round(counts/indx_len * 100, 2)
        stats.append(out)
    
    for stat in reversed(stats):
        print(stat)
    

def get_lindx_csv_stat(csv_file):
    bounds = [0, 12.5, 25, 50, 75, 100]
    n = len(bounds)-1

    df = pd.read_csv(csv_file, header=0, usecols=['lindx'])
    indexes = df['lindx']
    df_len = indexes.shape[0]

    stats = []
    for i in range(n):
        lbd, rbd = bounds[i], bounds[i+1]
        if i == n:
            lmask = indexes >= rbd
            rmask = indexes <= lbd
            bnd = f"[{lbd}, {rbd}]"
        else:
            lmask = indexes >= bounds[i]
            rmask = indexes < bounds[i+1]
            bnd = f"[{lbd}, {rbd})"
        
        counts = (lmask & rmask).sum()
        
        out = {}
        out['bounds'] = bnd
        out['counts'] = counts
        out['%'] = round(counts/df_len * 100, 2)
        stats.append(out)
    
    for stat in reversed(stats):
        print(stat)


# PROCESSING

def _get_calc_type(var):
    if var == 'prec':
        return 'sum'
    elif var in ['tmean', 'tmin', 'tmax']:
        return 'mean'
    else:
        raise ValueError("Unknown variable {}".format(var))


def _bulk_baseline_cropper(var, noData=None):
    kwargs = {}

    if noData is not None:
        kwargs['noData'] = noData

    _var = 'tavg' if var == 'tmean' else var
    for i in range(1, 13):
        src_file = os.path.join(CF.RASTER_DIR, 'baseline', f'wc2.0_30s_{_var}', f'wc2.0_30s_{_var}_{i:02}.tif')        
        out_file = os.path.join(CF.CROPPED_DIR, 'baseline', var, f'{var}_{i}.tif')

        raster_crop(out_file, src_file, **kwargs)


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


def _bulk_future_merge_cropper(var, period):
    from wclim_config import MODELS, PERIODS

    if period not in PERIODS:
        raise

    for model in MODELS:
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


def _bulk_future_algebra_by_model(var, period, model, cutline=True):
    print(f'PROCESSING: {var}_{period}_{model}')

    if period not in WC.PERIODS.keys():
        raise
    if model not in WC.MODELS:
        raise
    
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


def _get_title(var, period):
    titles = {
        'prec': 'Precipitation ',
        'tmax': 'Max Temperature ',
        'tmean': 'Mean Temperature ',
        'tmin': 'Min Temperature '
    }
    title = titles[var]

    if period == 'baseline':
        title += '(Near-Current)'
    elif period == '2030s':
        title += f'(2030)'
    elif period == '2050s':
        title += f'(2050)'

    label = 'mm' if var == 'prec' else 'Deg C.'

    return title, label


def main():
    var = 'prec'
    period = ['2030s', '2050s']

    # ----------------------
    # 1. CROPPING
    # ----------------------

    # 1a. BASELINE: Crop tiff
    # _bulk_baseline_cropper(var)

    # 1b. FUTURE: Merge tiff
    # _bulk_future_merge_cropper(var, period=period[0])


    # ----------------------
    # 2. RASTER ALGEBRA
    # ----------------------

    # 2a. BASELINE: Raster Algebra
    # _bulk_baseline_algebra(var)

    # 2b. FUTURE: Raster Algebra
    # i. Per Model
    # _period = period[0]
    # for model in WC.MODELS:
    #     _bulk_future_algebra_by_model(var, _period, model)

    # ii. Per Var
    # _period = period[1]
    # _bulk_future_algebra(var, _period)


    # ----------------------
    # 3. RASTER FILE STATISTICS
    # ----------------------

    # Get Raster Statistics
    # var = 'tmax'
    # period = 'baseline'
    # model = 'miroc_miroc5'
    # calc_type = _get_calc_type(var)
    # src_file = os.path.join(CF.CROPPED_DIR, period, var, model, f'{var}_{calc_type}.tif')
    # sample_tif =  os.path.join(CF.CROPPED_DIR, period, var, f'{var}_mean_{period}.tif')
    # sample_tif = r'F:\thesis\cacao_suitability\cropped\2030s\prec\bcc_csm1_1\prec_sum.tif'
    # sample_tif = r'F:\thesis\cacao_suitability\cropped\baseline\prec\prec_sum_baseline.tif'
    # raster_info(sample_tif)


    # ----------------------
    # 4. PLOTTING
    # ----------------------
    # NOTE: For Future Data aside from prec
    # set appy_factor = 0.1

    # var = 'prec'
    # period = 'baseline'
    # title, label = _get_title(var, period)
    # src_file = os.path.join(CF.CROPPED_DIR, period, var, f'{var}_sum_{period}.tif')
    # raster_plot(src_file, var, title=title, label=label)


    # ----------------------
    # 5. RASTER TO XYZ (For BASE CSV)
    # ----------------------
    # NOTE: For Future Data aside from prec AND not for base csv
    # set appy_factor = 0.1
    # src_file = r'F:\thesis\cacao_suitability\cropped\2030s\prec\prec_mean_2030s.tif'
    # output_file = None
    # raster_to_xyz(src_file, output_file=output_file, clipped=True)


    # ----------------------
    # 6. RASTERS TO XYZ (LAND INDEX)
    # ----------------------
    # period = 'baseline'
    # rasters_to_lidx_xyz(period)


    # ----------------------
    # 7. CREATE MODEL FROM LINDX
    # ----------------------


    # ----------------------
    # 8. LINDX TO TIF
    # ----------------------
    # csv_file =  # path here
    # sample_tif =  # path here
    # lindx_csv_to_tif(csv_file, sample_tif)


    # ----------------------
    # 9. LINDX STAT
    # ----------------------

    # get_lindx_csv_stat(csv_file)

    tif_file = r'F:\thesis\cacao_suitability\cropped\2050s\model_adam_swish_DFF_lindx.tif'
    # tif_file = r'F:\thesis\cacao_suitability\cropped\2050s\2050s_lindx.tif'
    get_lindx_tif_stat(tif_file)

    # ----------------------
    # 10. LINDX REPLOT
    # ----------------------
    # BASE
    period = 'baseline'
    # file = os.path.join(CF.CROPPED_DIR, period, f'{period}_lindx.tif')
    # # MODELED
    file = os.path.join(CF.CROPPED_DIR, period, f'model_adam_swish_DFF_lindx.tif')
    # PLOT
    # raster_plot(file, linx=True, title="Near-Current Suitability \n(Modeled)")

   



if __name__ == '__main__':
    main()