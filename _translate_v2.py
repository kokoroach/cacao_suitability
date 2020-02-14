import os
from config import ph_bounds

from subprocess import Popen, PIPE
from osgeo import gdal

try:
    target_dir = os.path.dirname(gdal.__file__)
    
    src_asc = r'_files\gcm\nimr_hadgem2_ao_rcp6_0_2080s_prec_30s_r1i1p1_no_tile_asc\prec_1.asc'
    out_tif = r'cropped\2030\prec\prec_1.asc'

    gdal.Translate(out_tif, src_asc, projWin=ph_bounds, format="GTiff")
    # cmd_tokens = f'gdal_translate -projwin {bounds} -of "GTiff" {src_asc} {out_tif}'
    
    # shell = Popen(cmd_tokens, cwd=target_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # out, err = shell.communicate()


except Exception as e:
    print(e)