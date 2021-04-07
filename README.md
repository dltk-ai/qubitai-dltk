# DLTK SDK
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)


[![DLTK Logo](dltk.png)](https://dltk.ai/)

## About

Our philosophy is to create a Deep Technologies platform with ethical AI for enterprises that offers meaningful insights and actions. 

DLTK Unified Deep Learning platform can be leveraged to build solutions that are Application-Specific and Industry-Specific where AI opportunity found by using DLTK SDKs, APIs and Microservices. With best of the breed AI Services from platform pioneers like H2O, Google's TensorFlow, WEKA and a few trusted open-sources models and libraries, we offer custom AI algorithms with co-innovation support. 

## Getting Started

### Pre-requisite

1. OpenDLTK : OpenDLTK is collection of open-source docker images, where processing of images, text or structured tabular data is done using state-of-the-art AI models.

Please follow the below link for instructions on [OpenDLTK Installation](https://docs.dltk.ai/getting_started/openDLTK_setup.html)

---

**Note**: To use third party AI engines please provide your credentials. Instructions on getting credentials and configuring are provided below.

---


### Installation

**Installing through pip**
```sh
    pip install qubitai-dltk
```

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

Choose any one of the above options for Installation

---

### Usage

```python
import dltk_ai
client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

text = "The product is very easy to use and has got a really good life expectancy."

sentiment_analysis_response = client.sentiment_analysis(text)

print(sentiment_analysis_response)
```

Important Parameters:

**1. API key:**


**2. base_url:**
The base_url is the url for the machine where base service is installed by _default_ its localhost, so base_url needs to be [http://localhost:8000]()

_Expected Output_
```json
{
  "spacy": {"emotion": "POSITIVE", "scores": {"neg": 0.0, "neu": 0.653, "pos": 0.347, "compound": 0.7496}}
}
```

---
## Services

**1. Machine Learning**

* ML Wrapper - It parse user request parameters

* ML Scikit - This Microservice uses widely used Scikit package for training and evaluating classification, regression, clustering models and other ML related tasks on dataset provided by user.

* ML H2O - This Microservice uses H2O.ai python SDK for training and evaluating classification, regression, clustering models and other ML related tasks on dataset provided by user.

* ML Weka - This Microservice uses WEKA for training and evaluating classification, regression, clustering models and other ML related tasks on dataset provided by user.

**2. NLP**

* This microservice provides features like Sentiment analysis, Name Entity Recognition, Tag Extraction using widely used ``Spacy`` and `NLTK` package. It also provide support for various AI engines like Azure & IBM.

**3. Computer Vision**

* CV Wrapper - This microservice receives images provided by user and route to right service based on the feature requested by them.

* Image Classification - This microservice classify images into various classes using pretrained model and also using supported AI Engines.

* Object Detection - This microservice detect objects in Images provided by user using pretrained model and using supported AI Engines.


## Reference

For more detail on DLTK features & usage please refer [DLTK SDK Client Documentation](https://docs.dltk.ai)

## License

The content of this project itself is licensed under [GNU LGPL, Version 3 (LGPL-3)](https://github.com/dltk-ai/qubitai-dltk/blob/master/LICENSE)

## Contact

QubitAI Email-ID - connect@qubitai.tech
