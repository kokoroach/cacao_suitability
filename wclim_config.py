
VARIABLES = {'prec':'prec',
             'tmax':'tmax',
             'tmean':'tmean',
             'tmin':'tmin'}

REGIONS = {'b5': 'b5',
           'b6': 'b6'}

PERIODS = {'2030s':'2030s',
           '2050s': '2050s'}

MODELS = [
    'bcc_csm1_1',
    # 'bcc_csm1_1_m',   # TODO: Check 2050, no tmean_2
    'cesm1_cam5',
    'csiro_mk3_6_0',
    'fio_esm',
    'gfdl_cm3',
    # 'gfdl_esm2g',   # TODO: Check 2030, no tmean_10
    'gfdl_esm2m',
    'giss_e2_h',
    'giss_e2_r',
    'ipsl_cm5a_lr',
    'miroc_esm',
    'miroc_esm_chem',
    'miroc_miroc5',
    'mohc_hadgem2_es',
    'mri_cgcm3',
    'ncar_ccsm4',
    'ncc_noresm1_m',
    'nimr_hadgem2_ao',
]

SCENARIO = 'rcp6_0'
FORMAT = 'asc'
RES = '30s'
ENSEMBLE = 'r1i1p1'

HOST = 'http://datacgiar.s3.amazonaws.com/ccafs/ccafs-climate/data/ipcc_5ar_ciat_tiled'
