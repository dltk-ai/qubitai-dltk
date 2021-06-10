from __future__ import print_function

import base64
import json
import time
from time import sleep, time

import requests

from dltk_ai.assertions import validate_parameters, is_url_valid, allowed_file_extension, hyper_parameter_check
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
        self.api_key = api_key
        self.base_url = base_url

    # Note: NLP functions

    def sentiment_analysis(self, text, sources=['nltk_vader'], **kwargs):
        """
        :param str text: The text on which sentiment analysis is to be applied.
        :param sources: algorithm to use for the analysis - azure/ibm_watson/spacy
        :kwargs reformat: reformat to a common format or not
        :return:
            obj:A json obj containing sentiment analysis response.
        """
        sources = [feature.lower() for feature in sources]
        supported_sources = ['nltk_vader', 'azure', 'ibm_watson']
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        assert text is not None and text != '', "Please ensure text is not empty"

        reformat = kwargs.get('reformat', True)
        body = {'text': text, 'sources': sources, 'reformat': reformat}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/sentiment/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        if response.status_code == 200:
            response = response.json()
        return response

    def pos_tagger(self, text, sources=['spacy'], **kwargs):
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
        reformat = kwargs.get('reformat', True)
        body = {'text': text, 'sources': sources, 'reformat': reformat}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/pos/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        if response.status_code == 200:
            response = response.json()
        return response

    def ner_tagger(self, text, sources=['spacy'], **kwargs):
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
        reformat = kwargs.get('reformat', True)
        body = {'text': text, 'sources': sources, 'reformat': reformat}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/ner/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        if response.status_code == 200:
            response = response.json()
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
        response = requests.post(url=url, data=body, headers=headers)
        if response.status_code == 200:
            response = response.json()
        return response

    def tags(self, text, sources=['rake'], **kwargs):
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
        reformat = kwargs.get('reformat', True)
        body = {'text': text, 'sources': sources, 'reformat': reformat}
        body = json.dumps(body)
        url = self.base_url + '/core/nlp/tags/'
        headers = {'ApiKey': self.api_key, 'Content-type': 'application/json'}
        response = requests.post(url=url, data=body, headers=headers)
        if response.status_code == 200:
            response = response.json()
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

    def check_cv_job_status(self, task_creation_response, wait_time=10):
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
                if time() - start_time > wait_time:
                    job_id = task_creation_response["job_id"]
                    print("It's taking too long than expected!!",
                          f"Use check_cv_job('{job_id}') to check status of your request")
                    break

        else:
            print(f"FAILED due to {task_creation_response.content}")
        return task_status_response

    def object_detection(self, image_url=None, image_path=None, object_detectors=['tensorflow'], output_types=["json"], reformat=True, wait_time=10):
        """
        This function is for object detection
        Args:
            output_types (list): Type of output requested by client: "json", "image"
            image_url: Image URL
            image_path: Local Image Path
            object_detectors: Supported object detectors ['tensorflow','azure']
            reformat: if True, reformat response to a common format, else not
            wait_time: wait time to get response

        Returns:
        Object_detection
        """
        output_types = [feature.lower() for feature in output_types]
        assert image_url is not None or image_path is not None, "Please choose either image_url or image_path"
        if image_url is not None:
            assert is_url_valid(image_url), "Enter a valid URL"
        supported_object_detectors = ['tensorflow', 'azure']
        assert len(supported_object_detectors) >= 1, f"Please select object_detectors from {supported_object_detectors}"

        assert len([classifier for classifier in object_detectors if classifier in supported_object_detectors]) > 0, f"Please select object_detectors from {supported_object_detectors}"

        assert "json" in output_types or "image" in output_types, "Please select at least 1 output type"
        assert type(wait_time) == int and 0 < wait_time <= 30, "Please provide a wait time in (0-30) seconds"

        object_detectors = {detector: True for detector in object_detectors}

        load = {
            "tasks": {"object_detection": True},
            "reformat": reformat,
            "configs": {
                "output_types": output_types,
                "object_detection_config": {
                    "tensorflow": object_detectors.get('tensorflow', False),
                    "azure": object_detectors.get('azure', False)
                }

            }
        }

        if image_url is not None and image_path is None:
            load["input_method"] = "image_url"
            load["image_url"] = image_url

        elif image_url is None and image_path is not None:
            allowed_extensions = ('.jpg', '.png', '.jpeg')
            assert allowed_file_extension(image_path, allowed_extensions), f"Supported Files extensions are {allowed_extensions}"
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                load["base64_img"] = base64_img
                load["input_method"] = "base64_img"

        headers = {'ApiKey': self.api_key}
        url = self.base_url + '/computer_vision/object_detection/'

        task_response = requests.post(url, json=load, headers=headers)
        response = self.check_cv_job_status(task_response, wait_time)

        return response

    def image_classification(self, image_url=None, image_path=None, top_n=3, image_classifiers=['tensorflow'],
                             output_types=["json"], reformat=True, wait_time=10):
        """
        This function is for image classification
        Args:

            top_n: get top n predictions
            output_types (list): Type of output requested by client: "json", "image"
            image_url: Image URL
            image_path: Local Image Path
            image_classifiers: Supported image classifier tensorflow, azure, ibm
            reformat: if True, reformat responses received from azure, ibm to a common format
            wait_time: wait time to get response from DLTK server

        Returns:
        Image classification response
        """
        output_types = [feature.lower() for feature in output_types]
        assert image_url is not None or image_path is not None, "Please choose either image_url or image_path"
        if image_url is not None:
            assert is_url_valid(image_url), "Enter a valid URL"

        supported_image_classifiers = ['tensorflow', 'azure', 'ibm']

        assert len(image_classifiers) >= 1, f"Please select image_classifiers from {supported_image_classifiers}"
        assert "json" in output_types or "image" in output_types, "Please select at least 1 output type"
        assert type(wait_time) == int and 0 < wait_time <= 30, "Please provide a wait time in (0-30) seconds"

        assert len([classifier for classifier in image_classifiers if classifier in supported_image_classifiers]) > 0, f"Please select image_classifiers from {supported_image_classifiers}"

        image_classifiers = {classifier: True for classifier in image_classifiers}

        load = {
            "tasks": {"image_classification": True},
            "reformat": reformat,
            "configs": {
                "output_types": ["json"],
                "img_classification_config": {
                    "top_n": top_n,
                    "tensorflow": image_classifiers.get('tensorflow', False),
                    "ibm": image_classifiers.get('ibm', False),
                    "azure": image_classifiers.get('azure', False)
                }
            }
        }

        if image_url is not None and image_path is None:
            load["input_method"] = "image_url"
            load["image_url"] = image_url

        elif image_url is None and image_path is not None:
            allowed_extensions = ('.jpg', '.png', '.jpeg')
            assert allowed_file_extension(image_path, allowed_extensions), f"Supported Files extensions are {allowed_extensions}"
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            load["base64_img"] = base64_img
            load["input_method"] = "base64_img"

        headers = {'ApiKey': self.api_key}
        url = self.base_url + '/computer_vision/image_classification'

        task_response = requests.post(url, json=load, headers=headers)
        response = self.check_cv_job_status(task_response, wait_time)

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
        assert allowed_file_extension(audio_path, '.wav'), f'Please use supported audio format {supported_audio_format}'
        assert all(i in supported_sources for i in sources), f"Please enter supported source {supported_sources}"
        body = {'audio': (audio_path, open(audio_path, 'rb'), 'multipart/form-data')}
        sources_string = ",".join(sources)
        payload = {'sources': sources_string}
        url = self.base_url + '/speech-to-text/'
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


    def get_default_params(self,service,algorithm):
        
        with open('dltk_ai\ml_hyperparameters.json') as file:
            hyperparameters = json.load(file)
    
        params = list(hyperparameters[service][algorithm].keys())
        default_values = [hyperparameters[service][algorithm][i]['default'] for i in list(hyperparameters[service][algorithm].keys())]
        return dict(zip(params,default_values))

    def train(self, service, algorithm, dataset, label, features, model_name=None, lib="weka", train_percentage=80, save_model=True, folds=5, cross_validation=False, params=None, dataset_source=None, evaluation_plots=False):

        """
        :param service: Training task to perform. Valid parameter values are classification, regression.
        :param algorithm: Algorithm used for training the model.
        :param dataset: dataset file location in DLTK storage.
        :param label: Target variable.
        :param features: List of features used for training the model.
        :param model_name: Model will be saved with the name specified in this parameter.
        :param lib: Library for training the model. Currently we are supporting scikit, h2o and weka.
        :param train_percentage: Percentage of data used for training the model. Rest of the data will be used to test the model.
        :param save_model: If True, the model will be saved in the DLTK Storage.
        :param dataset_source: To specify data source,
                None: Dataset file from DLTK storage will be used
                database: Query from connected database will be used
        :param folds: number of folds for cross validation
        :param cross_validation: Evaluates model using crossvalidation if set to True.
        :rtype: A json object containing the file path in storage.
        
        """


        service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(
            service, lib, algorithm, features, label, train_percentage)

        hyper_parameter_check(service,algorithm, params)

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
                    "params": params,
                    "folds" : folds,
                    "crossValidation" : cross_validation,
                    "evalPlots": evaluation_plots
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
                    "params": params,
                    "folds" : folds,
                    "crossValidation" : cross_validation,
                    "evalPlots": evaluation_plots
                }
            }
        body = json.dumps(body)
        response = requests.post(url=url, data=body, headers=headers)
        response = response.json()
        return response

    def feedback(self, service, algorithm, train_data, feedback_data, job_id, model_url, label, features, lib='weka',
                 model_name=None, split_perc=80,save_model=True, folds=5, cross_validation=False, params=None, evaluation_plots=False):
        """
        :param service: Training task to perform. Valid parameter values are classification, regression.
        :param algorithm: Algorithm used for training the model.
        :param train_data: dataset file location in DLTK storage.
        :param feedback_data: dataset file location in DLTK storage.
        :param job_id: job id from the train function used to train the model.
        :param model_url: model url returned from job output function.
        :param label: Target variable.
        :param features: List of features used for training the model.
        :param lib: Library for training the model. Currently we are supporting scikit, h2o and weka.
        :param model_name: Model will be saved with the name specified in this parameter.
        :param split_perc: Percentage of data to use for training the model. Rest of the data will be used to test the model.
        :param save_model: If True, the model will be saved in DLTK Storage.
        :param params: additional parameters.
        :param folds: number of folds for cross validation
        :param cross_validation: Evaluates model using crossvalidation if set to True.

        :rtype: A json object containing the file path in storage.
        """
        service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(
            service, lib, algorithm, features, label)
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
                'params': params,
                'saveModel': save_model,
                "folds" : folds,
                "crossValidation" : cross_validation,
                "evalPlots": evaluation_plots
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

    def face_analytics(self, image_url=None, features=None, image_path=None, face_detectors=['mtcnn'],
                       output_types=["json"], wait_time=10):
        """
        This function is for face analytics
        Args:
            output_types (list): Type of output requested by client: "json", "image"
            image_url: Image URL
            image_path: Local Image Path
            features: list of features requested by client
            face_detectors: supported face detectors are mtcnn, dlib, opencv, azure
            wait_time: wait time for server to return response

        Returns:
        face analytics response dependent on the features requested by client
        """
        if features is None:
            features = ['face_locations']
        features = [feature.lower() for feature in features]
        assert image_url is not None or image_path is not None, "Please choose either image_url or image_path"
        assert "json" in output_types or "image" in output_types, "Please select at least 1 output type ['json','image']"
        assert len(features)>0, "Please select at least one feature ['face_locations']"
        assert type(wait_time) == int and 0 < wait_time <= 30, "Please provide a wait time in (0-30) seconds"
        if 'face_locations' in features:
            if type(face_detectors) == str:
                face_detectors = face_detectors.split(',')
            face_detectors = [detector.lower() for detector in face_detectors]
            supported_face_detectors = ['mtcnn', 'azure', 'dlib', 'opencv']
            assert len([detector for detector in supported_face_detectors if detector in face_detectors]) > 0, f"Please choose face_detectors from {supported_face_detectors}"

        if image_url is not None:
            assert is_url_valid(image_url), "Enter a valid URL"

        face_detectors = {detector: True for detector in face_detectors}

        load = {
            "image_url": image_url,

            "tasks": {"face_detection": False},

            "configs": {
                "output_types": output_types,

                "face_detection_config": {
                    "dlib": face_detectors.get('dlib', False),
                    "opencv": face_detectors.get('opencv', False),
                    "mtcnn": face_detectors.get('mtcnn', False),
                    "azure": face_detectors.get('azure', False)
                }
            }
        }

        if 'face_locations' in features:
            load['tasks']['face_detection'] = True
        if image_url is not None and image_path is None:
            load["input_method"] = "image_url"
            load["image_url"] = image_url

        elif image_url is None and image_path is not None:
            allowed_extensions = ('.jpg', '.png', '.jpeg')
            assert allowed_file_extension(image_path, allowed_extensions), f"Supported Files extensions are {allowed_extensions}"
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            load["base64_img"] = base64_img
            load["input_method"] = "base64_img"

        headers = {'ApiKey': self.api_key}
        url = self.base_url + '/computer_vision/face_analytics/'

        task_response = requests.post(url, json=load, headers=headers)
        response = self.check_cv_job_status(task_response, wait_time)

        return response
