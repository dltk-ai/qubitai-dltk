# DLTK SDK (Python)
[![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/downloads/release/python-350/)


[![DLTK Logo](python/dltk.png)](https://dltk.ai/)

DLTK renders a comprehensive spectrum of solutions that can be accessed by users on-demand from our pool of transformational technologies.

### Installation

DLTK SDK requires Python 3.5 + . Go to https://dltk.ai/ and create an app. On creation of an app, you will get an API Key.

```sh
import dltk_ai
c = dltk_ai.DltkAiClient('API Key')
response = c.sentiment_analysis('I am feeling good.')
print(response)
```

For more details, visit https://dltk.ai/


## License

The content of this project itself is licensed under [GNU LGPL, Version 3 (LGPL-3)](https://github.com/dltk-ai/dltk-ai-sdk/blob/master/python/LICENSE)
