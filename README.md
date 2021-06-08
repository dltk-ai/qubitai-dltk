
<p align="center">
<a href="https://dltk.ai/">
  <img src="dltk.png" alt="DLTK logo">
</a>
</p>

<p align="center">
<a>
  <img src="https://img.shields.io/badge/python-3.8-blue.svg" alt="Python 3.8">
</a>

</p>



## About

Our philosophy is to create a Deep Technologies platform with ethical AI for enterprises that offers meaningful insights and actions. 

DLTK Unified Deep Learning platform can be leveraged to build solutions that are Application-Specific and Industry-Specific where AI opportunity found by using DLTK SDKs, APIs and Microservices. With best of the breed AI Services from platform pioneers like H2O, Google's TensorFlow, WEKA and a few trusted open-sources models and libraries, we offer custom AI algorithms with co-innovation support. 

## Getting Started

### Pre-requisite

* **OpenDLTK** : OpenDLTK is collection of open-source docker images, where processing of images, text or structured tabular data is done using state-of-the-art AI models.

   _Please follow the below link for instructions on [OpenDLTK Installation](https://docs.dltk.ai/getting_started/openDLTK_setup.html)_


### Installation

  **Installing through pip**
```sh
pip install qubitai-dltk
```

<br />

  **Installing from Source**

  1. Clone the repo

```sh
git clone https://github.com/dltk-ai/qubitai-dltk.git
``` 
  2. Set working directory to qubitai-dltk folder

```sh
cd qubitai-dltk
``` 

  3. Install requirements from requirements.txt file

```sh
pip install -r requirements.txt
```


<br />

## Usage

```python
import dltk_ai
client = dltk_ai.DltkAiClient(base_url='http://localhost:8000')

text = "The product is very easy to use and has got a really good life expectancy."

sentiment_analysis_response = client.sentiment_analysis(text)

print(sentiment_analysis_response)
```

**Important Parameters:**

> **APIkey** : a valid API key generated by following steps as shown [here](https://docs.dltk.ai/getting_started/generateAPIkey.html)
>
> **base_url** : The base_url is the url for the machine where base service is installed. (_default_: [http://localhost:8000]())

<br />

**Expected Output**
```json
{
  "nltk_vader": {"emotion": "POSITIVE", "scores": {"negative": 0.0, "neutral": 0.653, "positive": 0.347, "compound": 0.7496}}
}
```

<br>

## Services

#### Machine Learning
>
>**ML Scikit** -  This Microservice uses widely used Scikit package for training and evaluating classification, regression, clustering models and other ML related tasks on dataset provided by user.
>
>**ML H2O** - This Microservice uses H2O.ai python SDK for training and evaluating classification, regression, clustering models and other ML related tasks on dataset provided by user.
>
>**ML Weka** - This Microservice uses WEKA for training and evaluating classification, regression, clustering models and other ML related tasks on dataset provided by user.
>
>**Example Notebooks**
> - [ML Classification Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/machine_learning/DLTK%20ML%20Classification%20Tutorial.ipynb)
> - [ML Regrression Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/machine_learning/DLTK%20ML%20Regression%20Tutorial.ipynb)
> - [ML Clustering Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/machine_learning/DLTK%20ML%20Clustering%20Tutorial.ipynb)

<br/>

#### Natural Language Processing (NLP)
>
>* This microservice provides features like Sentiment analysis, Name Entity Recognition, Tag Extraction using widely used ``Spacy`` and `NLTK` package. It also provide support
> for various AI engines like Azure & IBM.
>
>**Example Notebook**
> - [NLP Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/natural_language_processing/DLTK%20NLP.ipynb)


<br/>

#### Computer Vision

>* **Image Classification** - This microservice classify images into various classes using pretrained model and also using supported AI Engines.
> 
>* **Object Detection** - This microservice detect objects in Images provided by user using pretrained model and using supported AI Engines.
>
> **Example Notebooks**
> - [Image Classification Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/computer_vision/DLTK%20Image%20Classification.ipynb)
> - [Object Detection Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/computer_vision/DLTK%20Object%20detection.ipynb)
> - [Face Analytics Colab Notebook](https://colab.research.google.com/github/dltk-ai/qubitai-dltk/blob/master/examples/computer_vision/DLTK%20Face%20Detection.ipynb)
>
> **Note**
> - To use third party AI engines like Microsoft Azure & IBM watson, please ensure that its credentials were configured while setting up openDLTK. 

<br/>

### Documentation

For more detail on DLTK features & usage please refer [DLTK SDK Client Documentation](https://docs.dltk.ai)

### License

The content of this project itself is licensed under [GNU LGPL, Version 3 (LGPL-3)](https://github.com/dltk-ai/qubitai-dltk/blob/master/LICENSE)


### Team

|[![](https://github.com/shreeramiyer.png?size=50)](https://github.com/shreeramiyer)| [![](https://github.com/GHub4Naveen.png?size=50)](https://github.com/GHub4Naveen) [![](https://github.com/alamcta.png?size=50)](https://github.com/alamcta) |[![](https://github.com/SivaramVeluri15.png?size=50)](https://github.com/SivaramVeluri15) [![](https://github.com/vishnupeesapati.png?size=49)](https://github.com/vishnupeesapati) [![](https://github.com/appareddyraja.png?size=50)](https://github.com/appareddyraja) [![](https://github.com/kavyavelagapudi252.png?size=50)](https://github.com/kavyavelagapudi252) [![](https://github.com/vivekkya.png?size=49)](https://github.com/vivekkya)
|:--:|:--:|:--:|
|Founding Member|Lead Maintainer|Core Contributor|

<br />

*For more details you can reach us at QubitAI Email-ID - [connect@qubitai.tech](connect@qubitai.tech)*
