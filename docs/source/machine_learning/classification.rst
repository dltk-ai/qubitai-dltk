Classification
================

Supported Libraries and Algorithms
----------------------------------

.. csv-table::
   :header: "scikit", "weka", "h2o"
   :widths: 30, 30, 30

   "NaiveBayesMultinomial", "LibSVM", "NaiveBayesBinomial"
   "LogisticRegression", "NaiveBayesMultinomial", "DeepLearning"
   "DecisionTrees","KStar"
   "Bagging","AdaBoostM1"
   "RandomForest","AttributeSelectedClassifier"
   "GradientBoostingMachines","Bagging"
   "XGradientBoosting","CostSensitiveClassifier"
   "AdaBoost","DecisionTable"



Training a model
-----------------

.. function:: client.train(service, algorithm, dataset, label, features, model_name=None,
                            lib="weka", train_percentage=80, save_model=True,params=None, 
                            dataset_source=None):

   :param service: Training task to perform. Valid parameter values are classification, regression.
   :param algorithm: Algorithm to use for training the model.
   :param dataset: dataset file location in DLTK storage.
   :param label: Target variable.
   :param features: List of features to use for training the model.
   :param model_name: Model will be saved with the name specified in this parameter.
   :param lib: Library for training the model. Currently we are supporting scikit, h2o and weka.
   :param train_percentage: % of data to use for training the model. Rest of the data will be used to test the model.
   :param save_model: If True, the model will be saved in DLTK Storage.
   :param dataset_source: To specify data source,
        None: Dataset file will from DLTK storage will be used
        database: Query from connected database will be used
   :rtype: A json obj containing the file path in storage.


Example::

    task = "classification"
    library = "weka"
    algorithm = "Logistic"
    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI", "DiabetesPedigreeFunction","Age"]
    label = 'Outcome'
    train_percentage = 80
    model_name = "DiabetesDetection"
    save_model = "true"
    
    train_response = client.train(task, algorithm, train_data, label,features,model_name, library, train_percentage, save_model)
    print(train_response)


Predictions
------------


.. function:: client.predict(service, dataset, model_url, features, lib='weka', 
                            params=None, dataset_source=None):
    
    :param service: Service used in training the model. Valid parameter values are classification, regression.
    :param dataset: dataset file location in DLTK storage.
    :param model_url: trained model location in DLTK storage.
    :param features: List of features used for training.
    :param lib: Library used for training the model. Currently we are supporting scikit, h2o and weka.
    :param dataset_source: To specify data source,
        None: Dataset file will from DLTK storage will be used
        database: Query from connected database will be used
    :rtype: A json obj containing the file info which has the predictions.

Example::

    task = "classification"
    library = "weka"
    test_data = '/dltk-ai/test_data.csv'
    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI", "DiabetesPedigreeFunction","Age"]
    model_url = '/dltk-ai/DiabetesDetection.mdl'
    
    prediction_response = client.predict(task, test_data, model_url, features, library)
    print(prediction_response)



Feedback
---------


.. function:: client.train(service, algorithm, train_data, feedback_data, job_id, model_url, 
                            label, features, lib='weka', model_name=None, 
                            split_perc=80, save_model=True, params=None):

   :param service: Training task to perform. Valid parameter values are classification, regression.
   :param algorithm: Algorithm to use for training the model.
   :param train_data: dataset file location in DLTK storage.
   :param feedback_data: dataset file location in DLTK storage.
   :param job_id:
   :param model_url:
   :param label: Target variable.
   :param features: List of features to use for training the model.
   :param lib: Library for training the model. Currently we are supporting scikit, h2o and weka.
   :param model_name: Model will be saved with the name specified in this parameter.
   :param split_perc: % of data to use for training the model. Rest of the data will be used to test the model.
   :param save_model: If True, the model will be saved in DLTK Storage.
   :param params: additional parameters.
   :rtype: A json obj containing the file path in storage.

Example::

    task = "classification"
    library = "weka"
    algorithm = "Logistic"
    train_data = '/dltk-ai/train_data.csv'
    feedback_data = '/dltk-ai/train_data.csv'
    job_id = '2457'
    model_url = '/dltk-ai/DiabetesDetection.mdl'
    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI", "DiabetesPedigreeFunction","Age"]
    label = 'Outcome'
    train_percentage = 80
    model_name = "DiabetesDetection"
    save_model = "true"
    
    feedback_response = client.feedback(task, algorithm, train_data, feedback_data, job_id, model_url,label, features, library, model_name, split_perc, save_model)
    print(feedback_response)

