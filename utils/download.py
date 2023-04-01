import os
import requests
import sys


def download(url, output_path=None):
    if output_path is None:
        return ValueError("Provide file path")

    if os.path.exists(output_path):
        return

    # Make sure parent directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    print('Downloading: {}'.format(url))
    with open(output_path, "wb") as f:
        r = requests.get(url, stream=True)
        r.raise_for_status()

        total_length = r.headers.get('content-length')
        if total_length is None:  # no content length header
            f.write(r.content)
        else:
            downloaded = 0
            total_length = int(total_length)

            for chuck in r.iter_content(chunk_size=1024):
                f.write(chuck)

                downloaded += len(chuck)
                _show_download_progress(total_length, downloaded)

    print('\nDownload successful!')
    return True


def _show_download_progress(total_length, downloaded):
    done = int(50 * downloaded / total_length)
    sys.stdout.write("\r[{}{}]".format('=' * done, ' ' * (50 - done)))
    sys.stdout.flush()
