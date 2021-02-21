import dltk_ai
from dltk_ai.dataset_types import Dataset

client = dltk_ai.DltkAiClient('YOUR_APIKEY')  # put your app key here.

# upload dataset - train
train_data_store_response = client.store('../examples/data/csv/housing_train.csv', Dataset.TRAIN_DATA)
print(train_data_store_response)
# get dataset url
train_data = train_data_store_response['fileUrl']

# upload dataset - test
test_file_store_response = client.store('../examples/data/csv/housing_test.csv', Dataset.TEST_DATA)
print(test_file_store_response)
# get dataset url
test_data = test_file_store_response['fileUrl']

# Model training using weka
# training params
task = "regression"
library = "weka"
algorithm = "LinearRegression"
label = "SalePrice"
features = ["LotShape", "Street"]
train_percentage = 80
save_model = "true"

# model training
train_response = client.train(task, algorithm, train_data, label, features,
                              "Housing Price Model", library, train_percentage, save_model)
print(train_response)

# check status of train job
# As training a model might take lot of time depending on size of dataset,
# we can check current status of model training using below functions

train_job_status_response = client.job_status(train_response['data']['jobId'])
print(train_job_status_response)

# Once model training job is finished (Status = "FINISH"), we can look into the model evaluation metrics
train_job_output_response = client.job_output(train_response['data']['jobId'])
print(train_job_output_response)

# get url where the model is saved, to use it for predictions
model = train_job_output_response['output']['modelUrl']

# predictions using the saved model on predictions data
predict_response = client.predict(task, test_data, features, model, library)
print(predict_response)

# check prediction job status
predict_job_status_response = client.job_status(predict_response['data']['jobId'])
print(predict_job_status_response)

# prediction output will be the dataset url of predictions file with predictions added as a column
predict_job_output_response = client.job_output(predict_response['data']['jobId'])
print(predict_job_output_response)
pred_file = predict_job_output_response['output']['predFileUrl']

# download the predictions file
prediction_response = client.download(pred_file)
print(prediction_response.text)

# feedback model - Regression
# Feedback params should be same as train params except for training percentage (can be changed accordingly)
# Job id, model url, dataset url used in training a model is required to feedback any model.
job_id = train_response['data']['jobId']

# IMP: Ensure the dataset has all features and label used for training the model.
# upload the feedback dataset
feedback_data_store_response = client.store('../examples/data/csv/housing_feedback.csv', Dataset.TRAIN_DATA)
print(feedback_data_store_response)

# upload the feedback dataset
feedback_data = feedback_data_store_response['fileUrl']

# initiate feedback training
feedback_response = client.feedback(task, algorithm, train_data, feedback_data, job_id, model,
                                    label, features, library, "Housing Price Model", 80, True)

print(feedback_response)

# get job status of feedback
feedback_job_status_response = client.job_status(job_id)
print(feedback_job_status_response)

# job output
feedback_job_output_response = client.job_output(job_id)
print(feedback_job_output_response)

# model url
model = train_job_output_response['output']['modelUrl']

# predict using the latest model trained with feedback data
feedback_predict_response = client.predict(task, test_data, features, model, library)
print(feedback_predict_response)

# get prediction job status
feedback_predict_job_status_response = client.job_status(predict_response['data']['jobId'])
print(feedback_predict_job_status_response)

# get prediction job response
feedback_predict_job_output_response = client.job_output(predict_response['data']['jobId'])
print(feedback_predict_job_output_response)

# get predictions file url and download
pred_file = feedback_predict_job_output_response['output']['predFileUrl']
response = client.download(pred_file)
print(response.text)
