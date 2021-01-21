import dltk_ai
from dltk_ai.dataset_types import Dataset


def main():
    c = dltk_ai.DltkAiClient('YOUR_APIKEY')  # put your app key here.
    # REGRESSION Training
    test_file_store_response = c.store('../csv/housing_test.csv', Dataset.TEST_DATA)
    print(test_file_store_response)
    test_data = test_file_store_response['fileUrl']
    train_data_store_response = c.store('../csv/housing_train.csv', Dataset.TRAIN_DATA)
    print(train_data_store_response)
    train_data = train_data_store_response['fileUrl']

    train_response = c.train("regression", "LinearRegression", train_data, "SalePrice", ["LotShape", "Street"],
                             "Housing Price Model", "weka", 80, True)  # this is the configuration.
    print(train_response)
    train_job_status_response = c.job_status(train_response['data']['jobId'])
    print(train_job_status_response)
    train_job_output_response = c.job_output(train_response['data']['jobId'])
    print(train_job_output_response)
    model = train_job_output_response['output']['modelUrl']
    predict_response = c.predict("regression", test_data, model, "weka")
    print(predict_response)
    predict_job_status_response = c.job_status(predict_response['data']['jobId'])
    print(predict_job_status_response)
    predict_job_output_response = c.job_output(predict_response['data']['jobId'])
    print(predict_job_output_response)
    pred_file = predict_job_output_response['output']['predFileUrl']
    prediction_response = c.download(pred_file)
    print(prediction_response.text)

    # REGRESSION Feedback
    job_id = train_response['data']['jobId']
    # IMP: Ensure the dataset has all features and label used for training the model.
    feedback_data_store_response = c.store('../csv/housing_feedback.csv', Dataset.TRAIN_DATA)
    print(feedback_data_store_response)
    feedback_data = feedback_data_store_response['fileUrl']

    feedback_response = c.feedback("regression", "LinearRegression", train_data, feedback_data, job_id, model,
                                   "SalePrice", ["LotShape", "Street"], "weka", "Housing Price Model", 80, True)

    print(feedback_response)
    feedback_job_status_response = c.job_status(job_id)
    print(feedback_job_status_response)
    feedback_job_output_response = c.job_output(job_id)
    print(feedback_job_output_response)
    model = train_job_output_response['output']['modelUrl']
    feedback_predict_response = c.predict("classification", test_data, model, "weka")
    print(feedback_predict_response)
    feedback_predict_job_status_response = c.job_status(predict_response['data']['jobId'])
    print(feedback_predict_job_status_response)
    feedback_predict_job_output_response = c.job_output(predict_response['data']['jobId'])
    print(feedback_predict_job_output_response)
    pred_file = feedback_predict_job_output_response['output']['predFileUrl']
    response = c.download(pred_file)
    print(response.text)


if __name__ == '__main__':
    main()
