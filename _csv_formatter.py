import csv
import os
import numpy as np

from osgeo import gdal
from linear_mapper import LinearMapper
from itertools import groupby
from decimal import Decimal

mapper = LinearMapper()


def format_csv(file_in, file_out, clim_type, interpolate=False):
    with open(file_in, 'r') as csvfile_in,\
        open(file_out, 'w', newline='') as csvfile_out:

        reader = csv.reader(csvfile_in, delimiter=' ')
        writer = csv.writer(csvfile_out, delimiter=',')
        
        for row in reader:
            z = row[-1]
            if z == 'Z':
                writer.writerow(row)
                continue
            val = float(z)
            if val < 0:
                continue
            if interpolate:
                row[-1] = mapper.get_linear_map(clim_type, val)
            writer.writerow(row)


def count_dry_months(file_in, file_out, prec_dir):
    with open(file_in, 'r') as csvfile_in,\
        open(file_out, 'w', newline='') as csvfile_out:

        reader = csv.reader(csvfile_in, delimiter=',')
        writer = csv.writer(csvfile_out, delimiter=',')
        
        rasters_files = []
        for dirpath, dirnames, filenames in os.walk(prec_dir):
            for file in filenames:
                ext = file.rsplit('.', 1)[-1]
                if 'mean' not in file and  ext == 'tif':
                    path = os.path.join(dirpath, file)
                    rasters_files.append(path)
            break
        
        rasters = []
        for raster in rasters_files:
            resp = get_raster_data(raster)
            rasters.append(resp)

        # rows = np.empty((0,3))
        for i, row in enumerate(reader):

            if i == 0:
                writer.writerow(row)
                continue
            
            # x, y = Decimal(row[0]), Decimal(row[1])
            coords = [float(row[0]), float(row[1])]

            cons_mnths = np.array([])
            for raster in rasters:
                gt = raster[0]
                data = raster[1]
                val = get_value_at_point(gt, data, coords)
                cons_mnths = np.append(cons_mnths, [bool(val)])
            
            if not np.any(cons_mnths):
                _cons_dry = 0
            else:
                _cons_dry = consecutive_one(cons_mnths)

            writer.writerow([row[0], row[1], _cons_dry])
        #     xyz = [[x, y ,_cons_dry]]
        #     rows = np.append(rows, xyz, axis=0)
            
            if i % 1000 == 0:
                print(i)
        #         writer.writerows(rows)
        #         rows = np.empty((0,3))

        # writer.writerows(rows)

def len_iter(items):
    return sum(1 for _ in items)

def consecutive_one(data):
    new_data = np.concatenate(([data[0]], data[:-1] != data[1:], [True]))
    result = np.diff(np.where(new_data)[0])[::2]
    return max(result)

    # return max(len_iter(run) for val, run in groupby(data) if val)


def get_src_file(self, file_name, src_dir):
    src_file = None
    
    try:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            for file in filenames:
                if file_name in file:
                    src_file = os.path.join(dirpath, file)
            break

        if src_file is None:
            raise Exception('File Not Found')
    
    except Exception as err:
        print(err)

    return src_file

def get_raster_data(raster):
    gdata = gdal.Open(raster)
    gt = gdata.GetGeoTransform()
    data = gdata.ReadAsArray().astype(np.float)
    gdata = None

    return gt, data 


def get_value_at_point(gt, data, pos):
    x = int((pos[0] - gt[0])/gt[1])
    y = int((pos[1] - gt[3])/gt[5])

    return data[y, x]


def for_format_csv():
    clim_type = 'prec'
    interpolate = True

    file_in = f'cropped\\baseline\\{clim_type}\\{clim_type}_mean.csv'
    file_out = f'cropped\\baseline\\{clim_type}\\{clim_type}_mean_r.csv'

    if interpolate:
        file_out = f'cropped\\baseline\\{clim_type}\\{clim_type}_mean_interp.csv'


    format_csv(file_in, file_out, clim_type, interpolate=interpolate)


def for_count_dry_months():
    clim_type = 'dry'

    file_in = f'cropped\\baseline\\prec\\csv\\prec_mean_r.csv'
    file_out = f'cropped\\baseline\\{clim_type}\\{clim_type}_count_r.csv'
    prec_dir = f'cropped\\baseline\\{clim_type}'

    count_dry_months(file_in, file_out, prec_dir)

def for_count_dry_months_2(append=False):
    clim_type = 'prec'

    file_in = f'cropped\\baseline\\{clim_type}\\csv\\{clim_type}_mean_r.csv'
    file_out = f'cropped\\baseline\\{clim_type}\\csv\\{clim_type}_01_dry_r.csv'
    prec_dir = f'cropped\\baseline\\prec'

    if append:
        count_dry_months_2_append(file_in, file_out, prec_dir)
    else:
        count_dry_months_2(file_in, file_out, prec_dir)




if __name__ == "__main__":
    # for_format_csv()
    for_count_dry_months()
    # for_count_dry_months_2()
    