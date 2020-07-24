import os
import sys
import requests

from wclim_config import *
from cacao_config import RASTER_DIR

def download_file(url, file_path=None):
    if file_path is None:
        return ValueError("Provide file path")

    if os.path.exists(file_path):
        print("WARNING! Check before overwrite for {}".format(file_path))
        return

    with open(file_path, "wb") as f:
        print('Downloading: {}'.format(url))

        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(r.content)
        else:
            downloaded = 0
            total_length = int(total_length)

            for chuck in r.iter_content(chunk_size=1024):
                f.write(chuck)

                downloaded += len(chuck)
                done = int(50 * downloaded / total_length)
                sys.stdout.write("\r[{}{}]".format('=' * done, ' '*(50-done)))    
                sys.stdout.flush()

    print('\nDownloaded: ', file_path)
    return file_path


def translate_option(var=None, region=None, period=None):
    var = VARIABLES[var]
    region = REGIONS[region]
    period = PERIODS[period]

    zipped_file = f'<MODEL>_{SCENARIO}_{period}_{var}_{RES}_{ENSEMBLE}_{region}_{FORMAT}.zip'
    url = f'{HOST}/{SCENARIO}/{period}/<MODEL>/{RES}/{zipped_file}'

    file_name = f'<MODEL>_{period}_{var}_{region}.zip'
    file_path = os.path.join(RASTER_DIR, period, var, file_name)

    return url, file_path


if __name__ == '__main__':
    option = {'var': 'tmean',
              'region': 'b5',
              'period': '2050s' }

    for model in MODELS:
        url, file_path = translate_option(**option)

        dl_url = url.replace('<MODEL>', model)
        file_path = file_path.replace('<MODEL>', model)

        download_file(dl_url, file_path=file_path)
