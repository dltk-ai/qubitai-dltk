# DLTK SDK
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)


[![DLTK Logo](dltk.png)](https://dltk.ai/)

## About

Our philosophy is to create a Deep Technologies platform with ethical AI for enterprises that offers meaningful insights and actions. 

DLTK Unified Deep Learning platform can be leveraged to build solutions that are Application-Specific and Industry-Specific where AI opportunity found by using DLTK SDKs, APIs and Microservices. With best of the breed AI Services from platform pioneers like H2O, Google’s TensorFlow, WEKA and a few trusted open-sources, we offer custom AI algorithms with co-innovation support. 

## Getting Started

### Pre-requisite

**QubitAI-OpenDLTK**

Please follow the below links for instructions

a. [Without Authentication](https://github.com/dltk-ai/openDLTK)

b. [With Authentication](https://github.com/dltk-ai/openDLTK/docs/auth.md)

---

**Note**: To use third party AI engines please provide your credentials. Instructions on getting credentials and configuring are provided below.

---


### Installation

**Installing from Source**

a. Clone the repo

```sh
   git clone https://github.com/dltk-ai/qubitai-dltk.git
``` 
b. Set working directory to qubitai-dltk folder

c. Install requirements from requirements.txt file

```sh
    pip install -r requirements.txt
```

**Installing through pip**
```sh
    pip install qubitai-dltk
```

---

### Usage

```python
import dltk_ai
client = dltk_ai.DltkAiClient('YOUR_API_KEY', base_url='http://localhost:8000')

text = "The product is very easy to use and has got a really good life expectancy."

sentiment_analysis_response = client.sentiment_analysis(text)

print(sentiment_analysis_response.text)
```

Important Parameters:

**1. API key:**
If authentication is disabled(default) in openDLTK server, there is no need to change 'YOUR_APIKEY' input, but if authentication is enabled you will need to provide a valid APIkey. 

For more details on authentication enabling, please refer to [Authentication Documentation](docs/auth.md)

**2. base_url:**
The base_url is the url for the machine where base service is installed by _default_ its localhost, so base_url needs to be [http://localhost:8000]()

_Expected Output_
```json
{
  "spacy": {"emotion": "POSITIVE", "scores": {"neg": 0.0, "neu": 0.653, "pos": 0.347, "compound": 0.7496}}
}
```

---

## Reference

For more detail on DLTK features & usage please refer [DLTK SDK Client Documentation](https://docs.dltk.ai)

## License

The content of this project itself is licensed under [GNU LGPL, Version 3 (LGPL-3)](https://github.com/dltk-ai/qubitai-dltk/blob/master/LICENSE)

## Contact

QubitAi - connect@qubitai.tech
