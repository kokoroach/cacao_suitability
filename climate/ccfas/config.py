from enum import Enum


class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class Climate(ExtendedEnum):
    """
    Different climate parameters considered for processing
    """
    PREC = 'prec'
    TMAX = 'tmax'
    TMEAN = 'tmean'
    TMIN = 'tmin'


class Period(ExtendedEnum):
    """
    TODO:
    """
    P_BASELINE = 'baseline'
    P_2030 = '2030s'
    P_2050 = '2050s'


class Region(ExtendedEnum):
    """
    TODO:
    """
    B5 = 'b5'
    B6 = 'b6'


class Model(ExtendedEnum):
    """
    TODO:
    """
    BCC_CSM1_1 = 'bcc_csm1_1'
    # BCC_CSM1_1_M = 'bcc_csm1_1_m'   # TODO: Check 2050, no tmean_2
    CESM1_CAM5 = 'cesm1_cam5'
    CSIRO_MK3_6_0 = 'csiro_mk3_6_0'
    FIO_ESM = 'fio_esm'
    GFDL_CM3 = 'gfdl_cm3'
    # GFDL_ESM2G = 'gfdl_esm2g'  # TODO: Check 2030, no tmean_10
    GFDL_ESM2M = 'gfdl_esm2m'
    GISS_E2_H = 'giss_e2_h'
    GISS_E2_R = 'giss_e2_r'
    IPSL_CM5A_LR = 'ipsl_cm5a_lr'
    MIROC_ESM = 'miroc_esm'
    MIROC_ESM_CHEM = 'miroc_esm_chem'
    MIROC_MIROC5 = 'miroc_miroc5'
    MOHC_HADGEM2_ES = 'mohc_hadgem2_es'
    MRI_CGCM3 = 'mri_cgcm3'
    NCAR_CCSM4 = 'ncar_ccsm4'
    NCC_NORESM1_M = 'ncc_noresm1_m'
    NIMR_HADGEM2_AO = 'nimr_hadgem2_ao'


# Based from https://ccafs-climate.org/data_spatial_downscaling/ with
# Extent: Region
# Format: ASCII Grid Format
# Period: {2030s, 2050s}
# Variable: as `class Climate`
# Resolution: 30seconds
# Scenario: RCP 6.0

DATA_URL = 'http://datacgiar.s3.amazonaws.com/ccafs/ccafs-climate/data/ipcc_5ar_ciat_tiled'  # noqa

SCENARIO = 'rcp6_0'
FORMAT = 'asc'
RESO = '30s'
ENSEMBLE = 'r1i1p1'
