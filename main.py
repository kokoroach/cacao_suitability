
from cacao.manager import (
    generate_raster_by_merge, generate_raster_by_crop,
    do_baseline_raster_calc, do_future_raster_calc,
    generate_future_calc_summary
)
from climate.ccfas.config import Climate, Period, Model
from climate.ccfas.downloader import download as download_ccfas
from climate.wclim.downloader import download as download_wclim


clim = Climate.PREC.value
period = Period.P_2030.value
model = Model.BCC_CSM1_1.value


# Utility func for future func invocation
def run_for_future(func):
    for _period in Period.values():
        if _period == Period.P_BASELINE.value:
            continue
        for _clim in Climate.values():
            func(_clim, _period)


# ----------------------
# 1. DOWNLOADING
# ----------------------

# Type: BASELINE
# 1a. Download Baseline Data
# download_wclim(clim)

# Type: FUTURE
# 1b. Download Future Data
# download_ccfas(clim, period)


# ----------------------
# 2. CROPPING
# ----------------------

# Type: BASELINE
# 2a.  Crop tiff
# generate_raster_by_crop(clim)

# Type: FUTURE
# 2b. Merged Crop tiff
# generate_raster_by_merge(clim, period)

# ----------------------
# 3. RASTER ALGEBRA CALCULATION
# ----------------------

# Type: BASELINE
# 3a. Raster algebra for baseline
# do_baseline_raster_calc(clim)

# Type: FUTURE
# 3b. Raster algebra for future
# do_future_raster_calc(clim, period)
# generate_future_calc_summary()
