"""A library that provides a Python interface to the DLTK APIs."""

__author__ = 'DLTK'
__email__ = 'connect@qubitai.tech'
__copyright__ = 'Copyright (c) 2019-2020 The QubitAI Technologies LLC'
__version__ = '1.0.8'
__url__ = 'https://github.com/dltk-ai/qubitai-dltk'
__download_url__ = ''
__description__ = 'A Python wrapper around the DLTK API'

from .core import DltkAiClient

import json
import requests
import warnings

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

try:
    url = f"https://pypi.org/pypi/qubitai-dltk/json"
    data = json.loads(requests.get(url).text)
    latest_version = data['info']['version']
    installed_version = __version__
    if str(installed_version) != str(latest_version):
        warnings.filterwarnings('ignore', '.*do not.*')
        warnings.formatwarning = custom_formatwarning
        warnings.warn(f'New version of dltk_ai ({latest_version}) available, you are still using older ({installed_version}) version of the dltk_ai, Please update using "pip install qubitai-dltk=={latest_version}"', FutureWarning)
except:
    pass
