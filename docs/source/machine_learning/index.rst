
*******************
Uploading a Dataset
*******************

Used to store file on cloud storage.

.. function:: client.store(file_path, dataset_type):

   :param file_path: The path of the dataset file.
   :param dataset_type: Type of dataset. valid Values TEST_DATA, TRAIN_DATA.
   :rtype: A json obj containing the file path in storage.

example::

    train_data_store_response = c.store('../train_data.csv',Dataset.TRAIN_DATA)
    print(train_data_store_response)


****************
Check Job Status
****************
After model training, a job is created with given details. It takes some time to train a model. To check the job status, we use the following function.

.. function:: client.job_status(job_id)

   :param job_id: jobId from the train function response.
   :rtype: A json obj containing the status details.

example::
   
    train_job_id = train_response['data']['jobId']
    train_job_status_response = client.job_status(train_job_id)
    print(train_job_status_response)

****************
Check Job Output
****************
Gives the output of training job which included model evaluation metrics, path where model is saved, etc.

.. function:: client.job_output(job_id)

   :param job_id: jobId from train function response.
   :rtype: A json obj containing the job output.

example::
   
    train_job_id = train_response['data']['jobId']
    train_job_output_response = client.job_output(train_job_id)
    print(train_job_output_response)

*****
Downloading a File
*****

Function used to download a file from cloud storage.

.. function:: client.download(file_url)

   :param file_url: url of file stored in cloud storage.
   :rtype: file content in text format.

example::
   
    prediction_file_url = predict_job_output_response['output']['predFileUrl']
    response = client.download(prediction_file_url)
    print(response)
