*****
About
*****

DLTK's Machine Learning service provides leverage to Train models, deploy them and use them for predictions, where the heavy lifting of training the models and deploying is done on DLTK's cloud server. The 3 widely used libraries (scikit, weka & h2o) for training models can be used under one platform.

*******************
Uploading a Dataset
*******************

Used to store file on cloud storage.

.. function:: client.store(file_path, dataset_type)

   :param file_path: The path of the dataset file.
   :param dataset_type: Type of dataset. valid Values TEST_DATA, TRAIN_DATA.
   :rtype: A json obj containing the file path in storage.

**Example**::

    import dltk_ai
    from dltk_ai.dataset_types import Dataset
    client = dltk_ai.DltkAiClient('YOUR_API_KEY')

    train_data_store_response = c.store('../train_data.csv',Dataset.TRAIN_DATA)
    print(train_data_store_response)


****************
Check Job Status
****************

Training a model is done on DLTK's cloud server. A job is triggered as soon as the train function is called. To check the status of the job, the following function can be used by giving ther job_id as input which is a response from train function.

.. function:: client.job_status(job_id)

   :param job_id: jobId from the train function response.
   :rtype: A json obj containing the status details.


****************
Check Job Output
****************
Gives the output of training job which includes model evaluation metrics, path where model is saved, etc.

.. function:: client.job_output(job_id)

   :param job_id: jobId from train function response.
   :rtype: A json obj containing the job output.


******************
Downloading a File
******************

Function used to download a file from cloud storage.

.. function:: client.download(file_url)

   :param file_url: url of file stored in cloud storage.
   :rtype: file content in text format.

