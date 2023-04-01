from cacao.config import WCLIM_DATA_DIR
from climate.wclim.config import DATA_URL, FILE_NAME
from climate.ccfas.config import Climate, Period
from utils.download import download as _download


def _get_filename(clim):
    return FILE_NAME.replace('^CLIMATE', clim)


def download(clim):
    url = generate_url(clim)
    output_path = get_output_file(clim)

    _download(url, output_path=output_path)


def generate_url(clim):
    if clim == Climate.TMEAN.value:
        clim = 'tavg'

    file_name = _get_filename(clim)
    url = f'{DATA_URL}/{file_name}'
    return url


def get_output_file(clim):
    file_name = _get_filename(clim)

    period = Period.P_BASELINE.value
    file_path = WCLIM_DATA_DIR / period / clim / file_name
    return file_path
