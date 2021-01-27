import dltk_ai

client = dltk_ai.DltkAiClient('YOUR_APIKEY')  # put your app key here.
# REGRESSION Training
test_file_store_response = client.store('../csv/rg_test.csv')
print(test_file_store_response)
test_data = test_file_store_response['fileUrl']
train_data_store_response = client.store('../csv/rg_train.csv')
print(train_data_store_response)
train_data = train_data_store_response['fileUrl']
train_response = client.train("classification", "LogisticRegression",  train_data, 'Revenue.Grid',['children', 'year_last_moved', 'Average.Credit.Card.Transaction'],"Revenue_Grid_Model", "scikit", 80, True)# this is the configuration.
print(train_response)
train_job_status_response = client.job_status(train_response['data']['jobId'])
print(train_job_status_response)
train_job_output_response = client.job_output(train_response['data']['jobId'])
print(train_job_output_response)
model = train_job_output_response['output']['modelUrl']
predict_response = client.predict("classification", test_data, model, "scikit", features=['children', 'year_last_moved', 'Average.Credit.Card.Transaction'])
print(predict_response)
predict_job_status_response = client.job_status(predict_response['data']['jobId'])
print(predict_job_status_response)
predict_job_output_response = client.job_output(predict_response['data']['jobId'])
print(predict_job_output_response)
pred_file = predict_job_output_response['output']['predFileUrl']
prediction_response = client.download(pred_file)
print(prediction_response.text)
