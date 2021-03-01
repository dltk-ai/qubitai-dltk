"""A library that provides a Python interface to the DLTK APIs."""

__author__ = 'DLTK'
__email__ = 'connect@qubitai.tech'
__copyright__ = 'Copyright (c) 2019-2020 The QubitAI Technologies LLC'
__version__ = '1.0.0'
__url__ = 'https://github.com/dltk-ai/qubitai-dltk'
__download_url__ = ''
__description__ = 'A Python wrapper around the DLTK API'

from .core import DltkAiClient
from .barcode_extraction import barcode_extractor

import json
import requests
import warnings

try:
    url = f"https://pypi.org/pypi/qubitai-dltk/json"
    data = json.loads(requests.get(url).text)
    latest_version = data['info']['version']
    installed_version = __version__
    if str(installed_version) != str(latest_version):
        warnings.filterwarnings('ignore', '.*do not.*')
        warnings.warn(f'New version of dltk_ai ({latest_version}) available, you are still using older ({installed_version}) version of the dltk_ai', FutureWarning)
except:
    pass
