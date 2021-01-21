from __future__ import print_function

import json
import time
import requests

from dltk_ai.dataset_types import Dataset


class DltkAiClient:
    """

        Attributes:
            api_key (str): API Key Generated for an app in DltkAi.
    """

    def __init__(self, api_key):
        """
            The constructor for DltkAi Client.

            Parameters:
                api_key (str): API Key Generated for an app in DltkAi.

            Returns:
                DltkAiClient: Client object for DltkAi.
        """
        self.api_key = api_key
        self.base_url = "https://prod-kong.dltk.ai"

    def sentiment_analysis(self, text):
        """
        :param str text: The text on which sentiment analysis is to be applied.
        :return:
            obj:A json obj containing sentiment analysis response.
        """

        body = {'text': text}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/sentiment/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        print(response.text)
        return response

    def sentiment_analysis_compare(self, text, sources):
        """
        :param str text: The text on which sentiment analysis is to be applied.
        :param sources: algorithm to use for the analysis - azure/ibm_watson/spacy
        :return:
            obj:A json obj containing sentiment analysis response.
        """

        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/sentiment/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        print(response.text)
        return response

    def pos_tagger(self, text):
        """
        :param str text: The text on which POS analysis is to be applied.
        :return
            obj: A json obj containing POS tagger response.
        """

        body = {'text': text}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/pos/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def pos_tagger_compare(self, text, sources):
        """
        :param str text: The text on which POS analysis is to be applied.
        :param sources: algorithm to use for POS analysis - ibm_watson/spacy
        :return:
            obj:A json obj containing POS analysis response.
        """

        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/pos/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        print(response.text)
        return response

    def ner_tagger(self, text):
        """
        :param str text: The text on which NER Tagger is to be applied.
        :return
            obj: A json obj containing NER tagger response.
        """

        body = {'text': text}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/ner/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def ner_tagger_compare(self, text, sources):
        """
        :param str text: The text on which NER Tagger is to be applied.
        "param sources: algorithm to use for NER Tagger - azure/ibm_watson/spacy
        :return:
            obj:A json obj containing NER Tagger response.
        """

        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/ner/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        print(response.text)
        return response

    def dependency_parser(self, text):
        """
        :param str text: The text on which Dependency Parser is to be applied.
        :return
            obj: A json obj containing dependency Parser response.
        """

        body = {'text': text}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/dependency-parser/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def tags(self, text):
        """
        :param str text: The text on which tags is to be applied.
        :return
            obj: A json obj containing tags response.
        """

        body = {'text': text}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/tags/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def tags_compare(self, text, sources):
        """
        :param str text: The text on which tags is to be applied.
        :param sources: algorithm to use for tagging - azure/ibm_watson/rake
        :return:
            obj:A json obj containing tags response.
        """

        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/tags/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        print(response.text)
        return response

    def face_detection_image(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            text : A base64 decoded image with face detected.
        """

        body = {'file': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/vision/face-detection/image'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers)
        return response.content

    def face_detection_image_core(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            text : A base64 decoded image with face detected.
        """

        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/vision/face-detection/image'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers)
        return response.content

    def face_detection_json(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            obj : A list of co-ordinates for all faces detected in the image.
        """
        body = {'file': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/vision/face-detection/json'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def face_detection_json_core(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            obj : A list of co-ordinates for all faces detected in the image.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/vision/face-detection/json'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def face_detection_compare(self, image_path, sources):
        """
        :param str image_path: The path of the image file.
        :param sources: algorithm to use for face detection - azure/opencv
        :return
            obj : A base64 decoded image with face detected.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        payload = {'sources': sources}
        url = self.base_url + '/core/vision/face-detection/compare'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers, data=payload)
        return response

    def eye_detection_image(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            text : A base64 decoded image with eye detected.
        """

        body = {'file': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/vision/eye-detection/image'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers)
        return response.content

    def eye_detection_json(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            obj : A list of co-ordinates for all eyes detected in the image.
        """
        body = {'file': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/vision/eye-detection/json'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def smile_detection_image(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            text : A base64 decoded image with smile detected.
        """

        body = {'file': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/vision/smile-detection/image'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers)
        return response.content

    def smile_detection_json(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            obj : A list of co-ordinates for all smiles detected in the image.
        """
        body = {'file': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/vision/smile-detection/json'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def object_detection_image(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            text : A base64 decoded image with objects detected.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/vision/object-detection/image'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers)
        return response.content

    def object_detection_json(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            obj : A list of co-ordinates for all objects detected in the image.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/vision/object-detection/json'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def object_detection_compare(self, image_path, sources):
        """
        :param str image_path: The path of the image file.
        :param sources: algorithm to use for object detection - tensorflow/azure
        :return
            obj : A base64 decoded image with objects detected.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        payload = {'sources': sources}
        url = self.base_url + '/core/vision/object-detection/image/compare'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers, data=payload)
        return response

    def image_classification(self, image_path):
        """
        :param str image_path: The path of the image file.
        :return
            obj : A list of all classes detected in the image.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/vision/image-classification'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def image_classification_compare(self, image_path, sources):
        """
        :param str image_path: The path of the image file.
        :param sources: algorithm to use for image classification - azure/ibm_watson/tensorflow
        :return
            obj : A list of all classes detected in the image.
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        payload = {'sources': sources}
        url = self.base_url + '/core/vision/image-classification/compare'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers, data=payload)
        return response

    def folder_upload(self, image_folder, folder_name):
        """
        :param str image_folder: list of image paths to be uploaded
        :param str folder_name: name of the folder for storing
        :return
            obj : A json obj containing list of image linkes stored.
        """
        files = []
        for i in range(len(image_folder)):
            files.append(('files', open(image_folder[i], 'rb')))
        url = self.base_url + '/s3/folder'
        payload = {}
        headers = {'label': folder_name, 'ApiKey': self.api_key}
        response = requests.post(url=url, headers=headers, data=payload, files=files).json()
        return response

    def visual_search(self, url1, url2):
        """
        :param str url1: url of a webpage
        :param str url2: url of other webpage
        :return
            obj : A json obj containing list source image string and matched image strings
        """
        body = {'url1': url1, 'url2': url2}
        payload = json.dumps(body)
        url = self.base_url + '/core/visual_search/'
        headers = {'ApiKey': self.api_key, 'Content-Type': 'application/json'}
        response = requests.post(url=url, data=payload, headers=headers).json()
        return response

    def helmet_detection(self, image_path):
        """
        :param str image_path: The path of the image file
        :return
            obj : A json object containing bounding boxes and image string
        """
        body = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/helmet-detection/'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, headers=headers, files=body).json()
        return response
    
    def serial_number_extraction(self, image_path, color):
        """
        :param str image_path: path of the image
        :param str color: B,G,R format color values in string
        :return
            obj : A json obj serial number, bounding box values and image string
        """
        body={'color': color}
        files = {'image': (image_path, open(image_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/core/vision/serial_number/'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, headers=headers, data=body, files=files).json()
        return response


    def speech_to_text(self, audio_path):
        """
        :param str audio_path: the path of the audio file.
        :return:
            obj: A json obj containing transcript of the audio file.
        """
        body = {'audio': (audio_path, open(audio_path, 'rb'), 'multipart/form-data')}
        url = self.base_url + '/speech-to-text/'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers).json()
        return response

    def speech_to_text_compare(self, audio_path, sources):
        """
        :param str audio_path: the path of the audio file.
        :param sources: algorithm to use for speech to text conversion - google/ibm_watson
        :return:
            obj: A json obj containing transcript of the audio file.
        """
        body = {'audio': (audio_path, open(audio_path, 'rb'), 'multipart/form-data')}
        payload = {'sources': sources}
        url = self.base_url + '/speech-to-text/compare'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers, data=payload).json()
        return response


    def train(self, service, algorithm, dataset_url, label, features, model_name=None, lib="weka", train_percentage=80,
              save_model=True, params=None):
        """
        :param lib: Library for training the model. Currently we are supporting DLTK and weka libraries.
        :param service: Valid parameter values are classification, regression.
        :param model_name: Model name and with this name model will be saved.
        :param algorithm: algorithm by which model will be trained.
        :param dataset_url: dataset file location in DLTK storage.
        :param label: label of the column in dataset file.
        :param train_percentage: % of data will be used for training and model will be tested against remaining % of data.
        :param features: column name list which is used to train classification model.
        :param save_model: If true model will saved in.
        :param params: additional parameters.
        :return:
            obj: A json obj containing model info.

        """
        url = self.base_url + '/machine/' + service + '/train/'
        headers = {"ApiKey": self.api_key, "Content-type": "application/json"}
        if params is None:
            params = {}
        if model_name is None:
            model_name = algorithm
        body = {
            "library": lib,
            "task": "train",
            "config": {
                "name": model_name,
                "algorithm": algorithm,
                "datasetUrl": dataset_url,
                "label": label,
                "trainPercentage": train_percentage,
                "features": features,
                "saveModel": save_model,
                "params": params
            }
        }
        body = json.dumps(body)
        response = requests.post(url=url, data=body, headers=headers)
        response = response.json()
        return response

    def feedback(self, service, algorithm, train_data, feedback_data, job_id, model_url, label, features, lib='weka',
                 model_name=None, split_perc=80, save_model=True, params=None):
        """
         The function call to feedback service in DLTK ML.

        :param lib: Trained model's library
        :param service: Trained model's service.
        :param model_name: Trained model's name.
        :param algorithm: Trained model's algorithm.
        :param train_data: Trained model's dataset url.
        :param feedback_data:
                a)Dataset (used for feedback) file location in DLTK storage.
                b)Feedback dataset upload. IMP: Please ensure the dataset has all features used for training the model.
        :param job_id: Job_id from training API response.
        :param model_url: Model file location in DLTK storage.
        :param label: Trained model's label.
        :param split_perc: % of data will be used for training and model will be tested against remaining % of data.
        :param features: Trained model's features.
        :param save_model:If true model will saved in.
        :param params: Additional parameters.
        :return:
            A json obj containing feedback model info.
        """

        url = self.base_url + '/machine/' + service + '/feedback'

        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}

        if params is None:
            params = {}
        if model_name is None:
            model_name = algorithm

        body = {
            'library': lib,
            'service': service,
            'task': 'FEEDBACK',
            'config': {
                'jobId': job_id,
                'name': model_name,
                'algorithm': algorithm,
                'datasetUrl': train_data,
                'feedbackDatasetUrl': feedback_data,
                'modelUrl': model_url,
                'label': label,
                'trainPercentage': split_perc,
                'features': features,
                'saveModel': save_model,
                'params': params
            }
        }
        body = json.dumps(body)
        response = requests.post(url=url, data=body, headers=headers)
        return response.json()

    def predict(self, service, dataset_url, model_url, lib='weka', params=None, features=[]):
        """
        :param lib: Library for training the model. Currently we are supporting DLTK and weka libraries.
        :param service: Valid parameter values are classification, regression.
        :param dataset_url: dataset file location in DLTK storage.
        :param model_url: trained model location in DLTK storage.
        :param params:
        :return:
            obj: A json obj containing the file info which has the predictions.
        """
        url = self.base_url + '/machine/' + service + '/predict'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        if params is None:
            params = {}
        body = {
            'library': lib,
            # 'service': service,
            'config': {
                'datasetUrl': dataset_url,
                'modelUrl': model_url,
                'params': params,
                'features': features
            }
        }
        body = json.dumps(body)
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def cluster(self, service, algorithm, dataset_url, features, lib='weka', number_of_clusters=2, model_name=None,
                save_model=True, params=None):
        """
        :param lib: Library for clustering the model. Currently we are supporting DLTK, weka, H2O, scikit-learn
                    libraries. Valid values for this parameter: DLTK, weka, h2o, scikit
        :param service: Valid parameter values are CLUSTER.
        :param model_name: Model name and with this name model will be saved.
        :param algorithm: algorithm by which model will be trained.
        :param epsilon: epsilon is algorithm specific constant.
        :param dataset_url: dataset file location in DLTK storage.
        :param features: column name list which is used to train classification model.
        :param number_of_clusters: the dataset will be clustered into number of clusters.
        :param save_model: If true model will saved
        :param params:
        :return:
            obj: A json obj containing model info.
        """
        url = self.base_url + '/machine/cluster/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        if params is None:
            params = {}
        if model_name is None:
            model_name = algorithm
        body = {
            'library': lib,
            'task': 'CLUSTER',
            'service': service,
            'config': {
                'name': model_name,
                'algorithm': algorithm,
                'datasetUrl': dataset_url,
                'numOfClusters': int(number_of_clusters),
                'epsilon': 0.1,
                'features': features,
                'saveModel': save_model,
                'params': params
            }
        }
        body = json.dumps(body)
        response = requests.post(url=url, data=body, headers=headers)
        response = response.json()
        return response

    def job_status(self, job_id):
        """
        :param job_id: jobId from the train api response.
        :return:
            obj: A json obj containing the status details.
        """
        url = self.base_url + '/machine/job/status?id={0}'.format(job_id)
        JOB_STATUS_CHECK_INTERVAL = 5
        STATE = 'state'
        headers = {'ApiKey': self.api_key}
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            response = response.json()
            while response[STATE] == 'RUN':
                time.sleep(JOB_STATUS_CHECK_INTERVAL)
                response = requests.get(url=url, headers=headers).json()
            if response[STATE] == 'FAIL':
                raise Exception('Prediction job failed!')
        else:
            raise Exception('Error while checking the status. Got ' + str(response.status_code))
        return response

    def job_output(self, job_id):
        """
        :param job_id: job id from the train api response.
        :return:
            obj: A json obj containing the job output details.
        """

        url = self.base_url + '/machine/output/findBy?jobId={0}'.format(job_id)
        headers = {'ApiKey': self.api_key}
        response = requests.get(url=url, headers=headers)
        return response.json()

    def store(self, file_path, dataset_type):
        """
        :param dataset_type: Type of dataset. valid Values TEST_DATA, TRAIN_DATA.
        :param file_path: The path of the dataset file.
        :return:
            obj: A json obj containing the file path in storage.
        """
        if not isinstance(dataset_type, Dataset):
            raise TypeError('dataset type must be an instance of Dataset Enum')

        url = self.base_url + '/s3/file'
        headers = {'ApiKey': self.api_key, 'label': dataset_type.value}
        files = {'file': open(str(file_path), 'rb')}
        response = requests.post(url=url, headers=headers, files=files, verify=False)
        response = response.json()
        return response

    def download(self, file_url):
        """
        :param file_url: location url of file stored in cloud storage.
        :return:
            txt:  file content in simple text format.
        """
        url = self.base_url + '/s3/file/download?url={0}'.format(file_url)
        headers = {'ApiKey': self.api_key}
        response = requests.get(url=url, headers=headers)
        return response
        