import os
import sys

try:
    import osgeo
    osgeo_dir = os.path.dirname(osgeo.__file__)
    script_dir = os.path.join(osgeo_dir, 'scripts')

    if not os.listdir(script_dir):
        raise Exception("GDAL scripts not found")
except Exception as exc:
    sys.exit(exc)


def doit(cmd):
    from subprocess import Popen, PIPE

    p = Popen(cmd, cwd=script_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if not err:
        print(out.decode("utf-8"))
    else:
        raise Exception("Execution error: \n{}".format(err.decode("utf-8")))


from osgeo import gdal


def gdal_compare(golden_file, new_file):
    cmd = "python gdalcompare.py {} {}".format(golden_file, new_file)
    doit(cmd)


def gdal_merge(out_filename, input_files, out_format="GTiff", ul_lr=None, a_nodata=None, quiet=False):
    if isinstance(a_nodata, float):
        a_nodata = int(a_nodata)

    ul_lr = " ".join(str(v) for v in ul_lr)
    input_files = " ".join(input_files)
    
    quiet = '' if quiet else '-v'
    
    cmd = "python gdal_merge.py -o {o} -of {of} -tap {q} -ul_lr {ul_lr} -a_nodata {a_nodata} {input_files}".format(
        o=out_filename, of=out_format, q=quiet, ul_lr=ul_lr, a_nodata=a_nodata, input_files=input_files)

    doit(cmd)


def gdal_calc(out_filename, raster_files, calc_type, no_data=None):
    from string import ascii_uppercase as UPPER

    rasters = []
    calc = []

    for i, raster in enumerate(raster_files):
        index = UPPER[i]
        rasters.append('-{} {}'.format(index, raster))
        calc.append('({index}*({index}>0))'.format(index=index))

    rasters = " ".join(rasters)

    if calc_type == 'sum':
        calc_str = '+'.join(calc)

    elif calc_type == 'mean':
        calc_str = '+'.join(calc)
        calc_str = '({})/{}'.format(calc_str, len(raster_files))
    else:
        raise ValueError("Unknown type %s" % calc_str)

    NoDataValue = ''
    if no_data is not None:
        NoDataValue = '--NoDataValue={}'.format(no_data)

    cmd = """python gdal_calc.py --outfile={out_filename} --calc="({calc_str})" {rasters} --co="COMPRESS=LZW" {no_data}""".format(
        out_filename=out_filename, calc_str=calc_str, rasters=rasters, no_data=NoDataValue)

    doit(cmd)
