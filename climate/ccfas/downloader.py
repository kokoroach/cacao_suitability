from cacao.config import CCFAS_DATA_DIR
from climate.ccfas.config import (
    FORMAT, SCENARIO, RESO, ENSEMBLE, DATA_URL, Region, Model)
from utils.download import download as _download


def download(clim, period, model=None):
    """
    Download future data from the CCAFS Climate Database
    """
    for region in Region.keys():
        if model:
            download_file(clim, period, model, region)
        else:
            # If model is not provided, download all models
            for _model in Model.keys():
                download_file(clim, period, _model, region)


def download_file(clim, period, model, region):
    url = generate_url(clim, period, model, region)
    output_path = get_output_file(clim, period, model, region)

    _download(url, output_path=output_path)


def generate_url(clim, period, model, region):
    zipped_file = f'{model}_{SCENARIO}_{period}_{clim}_{RESO}_{ENSEMBLE}_{region}_{FORMAT}.zip'  # noqa
    url = f'{DATA_URL}/{SCENARIO}/{period}/{model}/{RESO}/{zipped_file}'
    return url


def get_output_file(clim, period, model, region):
    file_name = f'{model}_{period}_{clim}_{region}.zip'
    file_path = CCFAS_DATA_DIR / period / clim / file_name
    return file_path
