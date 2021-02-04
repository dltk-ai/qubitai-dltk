# DLTK SDK (Python)
[![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/downloads/release/python-350/)


[![DLTK Logo](dltk.png)](https://cloud.dltk.ai/)

## About

Our philosophy is to create a Deep Technologies platform with ethical AI for enterprises that offers meaningful insights and actions. 

DLTK Unified Deep Learning platform can be leveraged to build solutions that are Application-Specific and Industry-Specific where AI opportunity found by using DLTK SDKs, APIs and Microservices. With best of the breed AI Services from platform pioneers like H2O, Google’s TensorFlow, WEKA and a few trusted open-sources, we offer custom AI algorithms with co-innovation support. 

## Getting Started

1. Creating a new Account on https://cloud.dltk.ai/ to start consuming services.

    a. Click on Sign Up and create a new account.

    b. A popup appears to enter the details, after entering the details click on "Sign Up Now" button.

    c. After creating your new account, enter your credentials to gain access

2. DLTK Cloud - Creating an App

    a. Once you login, to access DLTK Cloud, click on the Console button or the console tab available in the menu bar.

    b. Click on the Projects menu dropdown and select Create App

    c. Enter the name of the app and a short description

    d. The dashboard will be visible with the given name and description along with an API key

## Installation

## Installing from Source

a. Clone the repo

```sh
   git clone https://github.com/dltk-ai/qubitai-dltk.git
``` 
b. Set working directory to qubitai-dltk folder

c. Install requirements from requirements.txt file

```sh
    pip install -r requirements.txt
```

## Installing through pip
```sh
    pip install qubitai-dltk
```
### Usage

```sh
import dltk_ai
client = dltk_ai.DltkAiClient('API Key')
response = client.sentiment_analysis('I am feeling good.')
print(response)
```


## License

The content of this project itself is licensed under [GNU LGPL, Version 3 (LGPL-3)](https://github.com/dltk-ai/qubitai-dltk/blob/master/python/LICENSE)
