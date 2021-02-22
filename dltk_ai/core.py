from __future__ import print_function

import base64
import json
import time
from time import sleep, time

import requests

from dltk_ai.assertions import validate_parameters
from dltk_ai.dataset_types import Dataset


class DltkAiClient:
    """

        Attributes:
            api_key (str): API Key Generated for an app in DltkAi.
    """

    def __init__(self, api_key=None, base_url="https://prod-kong.dltk.ai"):
        """
            The constructor for DltkAi Client.

            Parameters:
                api_key (str): API Key Generated for an app in DltkAi.

            Returns:
                DltkAiClient: Client object for DltkAi.
        """
        assert api_key is not None, "Please provide a valid API key"
        self.api_key = api_key
        self.base_url = base_url

    # Note: NLP functions

    def sentiment_analysis(self, text, sources=['spacy']):
        """
        :param str text: The text on which sentiment analysis is to be applied.
        :param sources: algorithm to use for the analysis - azure/ibm_watson/spacy
        :return:
            obj:A json obj containing sentiment analysis response.
        """
        sources = [feature.lower() for feature in sources]
        supported_sources = ['spacy', 'azure', 'ibm_watson']
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        assert text is not None and text != '', "Please ensure text is not empty"

        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/sentiment/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)

        return response

    def pos_tagger(self, text, sources=['spacy']):
        """
        :param str text: The text on which POS analysis is to be applied.
        :param sources: algorithm to use for POS analysis - ibm_watson/spacy
        :return:
            obj:A json obj containing POS analysis response.
        """
        sources = [feature.lower() for feature in sources]
        supported_sources = ['spacy', 'ibm_watson']
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        assert text is not None and text != '', "Please ensure text is not empty"
        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/pos/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        return response

    def ner_tagger(self, text, sources=['spacy']):
        """
        :param str text: The text on which NER Tagger is to be applied.
        :param sources: algorithm to use for NER Tagger - azure/ibm_watson/spacy
        :return:
            obj:A json obj containing NER Tagger response.
        """
        sources = [feature.lower() for feature in sources]
        supported_sources = ['spacy', 'azure', 'ibm_watson']
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        assert text is not None and text != '', "Please ensure text is not empty"
        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/ner/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        return response

    def dependency_parser(self, text):
        """
        :param str text: The text on which Dependency Parser is to be applied.
        :return
            obj: A json obj containing dependency Parser response.
        """

        assert text is not None and text != '', "Please ensure text is not empty"
        body = {'text': text}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/dependency-parser/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def tags(self, text, sources=['rake']):
        """
        :param str text: The text on which tags is to be applied.
        :param sources: algorithm to use for tagging - azure/ibm_watson/rake
        :return:
            obj:A json obj containing tags response.
        """
        sources = [feature.lower() for feature in sources]
        supported_sources = ['rake', 'azure', 'ibm_watson']
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        assert text is not None and text != '', "Please ensure text is not empty"
        body = {'text': text, 'sources': sources}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/tags/compare'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)

        return response

    # Note: Computer vision functions
    def check_cv_job(self, job_id):
        """
               This function check status of the job
               Args:
                   job_id: job_id from check_cv_job_status will be returned if its taking too long

               Returns:
                   task_status_response: contains the result requested by the cv methods (only used when cv_job_status
                    crosses 10 second threshold)

               """

        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        task_url = f"{self.base_url}/computer_vision/task?task_id="
        task_status_response = requests.get(task_url + str(job_id), headers=headers).json()
        return task_status_response

    def check_cv_job_status(self, task_creation_response):
        """
        This function check status of the job
        Args:
            task_creation_response: response from task creation function

        Returns:

        """
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        task_url = f"{self.base_url}/computer_vision/task?task_id="
        task_status_response = {}
        # ensure the request was successful
        if task_creation_response.status_code == 200:
            task_creation_response = task_creation_response.json()
            task_status = "PENDING"

            start_time = time()
            while task_status != "SUCCESS":
                task_status_response = requests.get(task_url + task_creation_response['job_id'], headers=headers).json()
                task_status = task_status_response["task_status"]
                # wait for some time before checking status of the job again
                sleep(1)

                # check if execution time is more than
                if time() - start_time > 10:
                    print("It's taking too long than expected!!",
                          'you can use this function to check status dltkai.check_cv_job(',
                          task_creation_response['job_id'], ')')
                    break

        else:
            print(f"FAILED due to {task_creation_response.content}")
        return task_status_response

    def object_detection(self, image_url=None, image_path=None, tensorflow=True, azure=False, output_types=["json"]):
        """
        This function is for object detection
        Args:
            output_types (list): Type of output requested by client: "json", "image"
            image_url: Image URL
            image_path: Local Image Path
            tensorflow: if True, uses tensorflow for object detection
            azure: if True, returns azure results of object detection on given image

        Returns:
        Object_detection
        """
        output_types = [feature.lower() for feature in output_types]
        assert image_url is not None or image_path is not None, "Please choose either image_url or image_path"
        assert tensorflow is True or azure is True, "please choose at least 1 supported processor ['tensorflow', 'azure']"
        assert "json" in output_types or "image" in output_types, "Please select at least 1 output type"

        load = {
            "tasks": {"object_detection": True},
            "configs": {
                "output_types": output_types,
                "object_detection_config": {
                    "tensorflow": tensorflow,
                    "azure": azure
                }

            }
        }

        if image_url is not None and image_path is None:
            load["input_method"] = "image_url"
            load["image_url"] = image_url

        elif image_url is None and image_path is not None:
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                load["base64_img"] = base64_img
                load["input_method"] = "base64_img"

        headers = {'ApiKey': self.api_key}
        url = self.base_url + '/computer_vision/object_detection/'

        task_response = requests.post(url, json=load, headers=headers)
        response = self.check_cv_job_status(task_response)

        return response

    def image_classification(self, image_url=None, image_path=None, top_n=3, tensorflow=True, azure=False, ibm=False,
                             output_types=["json"]):
        """
        This function is for image classification
        Args:

            top_n: get top n predictions
            output_types (list): Type of output requested by client: "json", "image"
            image_url: Image URL
            image_path: Local Image Path
            tensorflow: if True, uses tensorflow for image classification
            azure: if True, returns azure results of image classification on given image
            ibm: if True, returns ibm results of image classification on given image

        Returns:
        Image classification response
        """
        output_types = [feature.lower() for feature in output_types]
        assert image_url is not None or image_path is not None, "Please choose either image_url or image_path"
        assert tensorflow is True or azure is True or ibm is True, "please choose at least 1 supported processor ['tensorflow', 'azure','ibm']"
        assert "json" in output_types or "image" in output_types, "Please select at least 1 output type"

        load = {
            "tasks": {"image_classification": True},

            "configs": {
                "output_types": ["json"],
                "img_classification_config": {
                    "top_n": top_n,
                    "tensorflow": tensorflow,
                    "ibm": ibm,
                    "azure": azure
                }
            }
        }

        if image_url is not None and image_path is None:
            load["input_method"] = "image_url"
            load["image_url"] = image_url

        elif image_url is None and image_path is not None:
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            load["base64_img"] = base64_img
            load["input_method"] = "base64_img"

        headers = {'ApiKey': self.api_key}
        url = self.base_url + '/computer_vision/image_classification'

        task_response = requests.post(url, json=load, headers=headers)
        response = self.check_cv_job_status(task_response)

        return response

    # Note: Speech Processing function

    def speech_to_text(self, audio_path, sources=['google']):
        """
        :param str audio_path: the path of the audio file.
        :param sources: algorithm to use for speech to text conversion - google/ibm_watson
        :return:
            obj: A json obj containing transcript of the audio file.
        """
        sources = [feature.lower() for feature in sources]
        supported_audio_format = '.wav'
        supported_sources = ['google', 'ibm_watson']
        assert '.wav' in audio_path, f'Please use supported audio format {supported_audio_format}'
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        body = {'audio': (audio_path, open(audio_path, 'rb'), 'multipart/form-data')}
        sources_string = ",".join(sources)
        payload = {'sources': sources_string}
        url = self.base_url + '/speech-to-text/compare'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, files=body, headers=headers, data=payload).json()
        return response

    # Note: ML functions

    def get_query_list(self):

        url = self.base_url + '/datasource/queries'
        headers = {'ApiKey': self.api_key}
        response = requests.post(url=url, headers=headers)
        if response.status_code == 200:
            response = response.json()
        else:
            raise Exception('Error while checking the query list. Got ' + str(response.status_code))
        return response

    def train(self, service, algorithm, dataset, label, features, model_name=None, lib="weka", train_percentage=80,
              save_model=True, params=None, dataset_source=None):
        """
        :param lib: Library for training the model. Currently we are supporting DLTK and weka libraries.
        :param service: Valid parameter values are classification, regression.
        :param model_name: Model name and with this name model will be saved.
        :param algorithm: algorithm by which model will be trained.
        :param dataset: dataset file location in DLTK storage.
        :param label: label of the column in dataset file.
        :param train_percentage: % of data will be used for training and model will be tested against remaining % of data.
        :param features: column name list which is used to train classification model.
        :param save_model: If true model will saved in.
        :param params: additional parameters.
        :return:
            obj: A json obj containing model info.

        Args:
            features: Feature list used while model training
            dataset_source: To specify data source,
                None: Dataset file will from DLTK storage will be used
                database: Query from connected database will be used

        """
        service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(
            service, lib, algorithm, features, label, train_percentage,
            save_model)

        url = self.base_url + '/machine/' + service + '/train/'
        headers = {"ApiKey": self.api_key, "Content-type": "application/json"}
        if params is None:
            params = {}
        if model_name is None:
            model_name = algorithm

        if dataset_source == "database":
            body = {
                "library": lib,
                "task": "train",
                "jobType": "DATABASE",
                "queryId": dataset,
                "config": {
                    "name": model_name,
                    "algorithm": algorithm,
                    "label": label,
                    "trainPercentage": train_percentage,
                    "features": features,
                    "saveModel": save_model,
                    "params": params
                }
            }
        else:
            body = {
                "library": lib,
                "task": "train",
                "config": {
                    "name": model_name,
                    "algorithm": algorithm,
                    "datasetUrl": dataset,
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
        service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(
            service, lib, algorithm, features, label,
            save_model)
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

    def predict(self, service, dataset, model_url, features, lib='weka', params=None, dataset_source=None):
        """
        :param lib: Library for training the model. Currently we are supporting DLTK and weka libraries.
        :param service: Valid parameter values are classification, regression.
        :param dataset: dataset file location in DLTK storage.
        :param model_url: trained model location in DLTK storage.
        :param features: list of features used for training
        :param params:
        :return:
            obj: A json obj containing the file info which has the predictions.
        
        Args:
            features: Feature list used while model training
            dataset_source: To specify data source,
                None: Dataset file will from DLTK storage will be used
                database: Query from connected database will be used

        """
        service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(
            service, lib, algorithm=None,
            features=features,
            label=None,
            cluster=True,
            predict=True)
        url = self.base_url + '/machine/' + service + '/predict'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        if params is None:
            params = {}
        if dataset_source == "database":
            body = {
                'library': lib,
                "jobType": "DATABASE",
                "queryId": dataset,
                # 'service': service,
                'config': {
                    'modelUrl': model_url,
                    'params': params,
                    'features': features
                }
            }
        else:
            body = {
                'library': lib,
                # 'service': service,
                'config': {
                    'datasetUrl': dataset,
                    'modelUrl': model_url,
                    'params': params,
                    'features': features
                }
            }
        body = json.dumps(body)
        response = requests.post(url=url, data=body, headers=headers).json()
        return response

    def cluster(self, service, algorithm, dataset, features, lib='weka', number_of_clusters=2, model_name=None,
                save_model=True, params=None, dataset_source=None):
        """
        :param lib: Library for clustering the model. Currently we are supporting DLTK, weka, H2O, scikit-learn
                    libraries. Valid values for this parameter: DLTK, weka, h2o, scikit
        :param service: Valid parameter values are CLUSTER.
        :param model_name: Model name and with this name model will be saved.
        :param algorithm: algorithm by which model will be trained.
        :param dataset: dataset file location in DLTK storage.
        :param features: column name list which is used to train classification model.
        :param number_of_clusters: the dataset will be clustered into number of clusters.
        :param save_model: If true model will saved
        :param dataset_source : metabase address for dataset
        :param params:
        :return:
            obj: A json obj containing model info.

        Args:
            dataset_source:
            dataset_source:
            features: Feature list used while model training
            dataset_source: To specify data source,
                None: Dataset file will from DLTK storage will be used
                database: Query from connected database will be used
        """
        service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(
            service, lib, algorithm, features,
            save_model, cluster=True)
        url = self.base_url + '/machine/cluster/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        if params is None:
            params = {}
        if model_name is None:
            model_name = algorithm
        if dataset_source == "database":
            body = {
                'library': lib,
                'task': 'CLUSTER',
                'service': service,
                "jobType": "DATABASE",
                "queryId": dataset,
                'config': {
                    'name': model_name,
                    'algorithm': algorithm,
                    'numOfClusters': int(number_of_clusters),
                    'epsilon': 0.1,
                    'features': features,
                    'saveModel': save_model,
                    'params': params
                }
            }
        else:
            body = {
                'library': lib,
                'task': 'CLUSTER',
                'service': service,
                'config': {
                    'name': model_name,
                    'algorithm': algorithm,
                    'datasetUrl': dataset,
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
                sleep(JOB_STATUS_CHECK_INTERVAL)
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

    # Todo: debug
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

    def face_analytics(self, image_url=None, features=None, image_path=None, dlib=False, opencv=True,
                       azure=False, mtcnn=False,
                       output_types=["json"]):
        """
        This function is for face analytics
        Args:
            output_types (list): Type of output requested by client: "json", "image"
            image_url: Image URL
            image_path: Local Image Path
            features (list) : Type of features requested by client
            dlib: if True, uses dlib for face analytics
            opencv: if True, uses opencv for face analytics
            azure: if True, returns azure results of face analytics on given image
            mtcnn: if True, uses mtcnn for face analytics

        Returns:
        face analytics response dependent on the features requested by client
        """
        if features is None:
            features = ['face_locations']
        features = [feature.lower() for feature in features]
        assert image_url is not None or image_path is not None, "Please choose either image_url or image_path"
        assert any(
            (azure, mtcnn, dlib, opencv)), "please choose at least 1 processor ['opencv', 'azure', 'mtcnn', 'dlib']"
        assert "json" in output_types or "image" in output_types, "Please select at least 1 output type ['json','image']"
        assert "face_locations" in features, "Please select at least one feature ['face_locations']"
        load = {
            "image_url": image_url,

            "tasks": {"face_detection": False},

            "configs": {
                "output_types": output_types,

                "face_detection_config": {
                    "dlib": dlib,
                    "opencv": opencv,
                    "mtcnn": mtcnn,
                    "azure": azure
                }
            }
        }

        if 'face_locations' in features:
            load['tasks']['face_detection'] = True
        if image_url is not None and image_path is None:
            load["input_method"] = "image_url"
            load["image_url"] = image_url

        elif image_url is None and image_path is not None:
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            load["base64_img"] = base64_img
            load["input_method"] = "base64_img"

        headers = {'ApiKey': self.api_key}
        url = self.base_url + '/computer_vision/face_analytics/'

        task_response = requests.post(url, json=load, headers=headers)
        response = self.check_cv_job_status(task_response)

        return response
