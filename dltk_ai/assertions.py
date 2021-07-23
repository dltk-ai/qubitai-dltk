import re
import json
import numpy as np
import os
from dltk_ai.ml_hyperparameters import hyperparameter_dictionary

supported_algorithm = {'regression':

                           {'scikit': ['LinearRegression', 'DecisionTrees', 'Bagging', 'RandomForest',
                                       'GradientBoostingMachines', 'XGradientBoosting', 'AdaBoost', 'ExtraTrees',
                                       'SupportVectorMachines'],
                            'h2o': ['LinearRegression', 'GradientBoostingMachines', 'RandomForest',
                                    'GradientBoostingMachine'],
                            'weka': ['LinearRegression', 'AdditiveRegression']
                            },

                       'classification': {
                           'scikit': ['NaiveBayesMultinomial', 'LogisticRegression', 'DecisionTrees', 'Bagging',
                                      'RandomForest', 'GradientBoostingMachines', 'XGradientBoosting', 'AdaBoost',
                                      'ExtraTrees', 'SupportVectorMachines', 'KNearestNeighbour'],
                           'h2o': ['NaiveBayesBinomial', 'DeepLearning'],
                           'weka': ['LibSVM', 'NaiveBayesMultinomial', 'KStar', 'AdaBoostM1',
                                    'AttributeSelectedClassifier', 'Bagging', 'DecisionTable', 'RandomTree', 'SMO',
                                    'Logistic', 'MultilayerPerceptron','RandomForest']
                       },

                       'clustering': {
                           'scikit': ['AgglomerativeClustering', 'KMeansClustering', 'MiniBatchKMeans', 'Birch',
                                      'DBScan', 'SpectralClustering'],
                           'weka': ['SimpleKMeans']
                       }
                       }


def validate_parameters(service: str, library: str, algorithm: str, features: list, label: str,
                        train_percentage: int = 0.80,
                        cluster=False, predict=False):
    # check service names
    allowed_service = ['regression', 'classification', 'clustering']
    service = service.lower()
    assert service in allowed_service, f"Please select *service* from {allowed_service}"

    # Check supported libraries
    allowed_libraries = ['scikit', 'h2o', 'weka']
    library = library.lower()
    assert library in allowed_libraries, f"Please select *Library* from {allowed_libraries}"

    # Check algorithm name & whether its supported or not
    if predict is False:
        assert algorithm in supported_algorithm[service][
            library], f"Unsupported Algorithm={algorithm} for {library} library ,\n please choose from these {supported_algorithm[service][library]}"
    if cluster is False:
        # Check train_percentage in range of (0,100)
        assert 0 < train_percentage <= 100, "please ensure train percentage is in range of 0-100, preferably inbetween 70-85%"

    # Check whether features list is not empty
    assert len(features) > 0, "Please ensure features is not an empty list, select atleast 1 feature"

    return service, library, algorithm, features, label, train_percentage


def validate_dataset(dataset_df, service, library, algorithm, features, target_variable):
    # check for Null Values
    null_count_df = dataset_df.isnull().sum(axis=0)

    assert null_count_df.sum() == 0, print(
        "please ensure there are no null values in your dataset \nNull Values count per column\n\n", null_count_df)

    # Check whether features & target Variables are present in dataset_df or not
    for feature in features:
        assert feature in list(dataset_df.columns), f"*{feature}* not found in Dataset column names"


def is_url_valid(string):
    """
    This function checks whether input string follow URL format or not
    Args:
        string: Input string

    Returns:
        True: if string follows URL format
        False: if string doesn't follow URL format

    >>> is_url_valid("C:/users/sample/Desktop/image.jpg")
    False
    >>> is_url_valid("/examples/data/image.jpg")
    False
    >>> is_url_valid("http://google.com")
    True
    >>> is_url_valid("https://images.financialexpress.com/2020/01/660-3.jpg")
    True
    >>> is_url_valid("https://images.financialexpress.com 2020/01/660-3.jpg")
    False

    """
    match_result = False

    pattern = re.compile(
        r'^(?:http|ftp)s?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(pattern, string):
        match_result = True
    return match_result


def allowed_file_extension(file_path, allowed_extensions):
    """
    To check whether file_path ends with allowed extensions or not
    Args:
        file_path: file path string
        allowed_extensions: set of extensions

    Returns:
        True: If file has a valid extension
        False: If file don't have valid extension

    >>> allowed_file_extension('sample.jpeg', ('.jpg', '.png', '.JPEG'))
    True
    >>> allowed_file_extension('sample.pdf', '.txt')
    False
    >>> allowed_file_extension('examples/sample.mp3', '.wav')
    False
    >>> allowed_file_extension('examples/sample.wav', '.wav')
    True
    """
    allowed_extensions = tuple([extension.lower() for extension in allowed_extensions])
    return file_path.lower().endswith(allowed_extensions)


def hyper_parameter_check(library, service, algorithm, user_input_params):

    try:

        user_input = list(user_input_params.keys())
        algorithm_params = hyperparameter_dictionary[library][service][algorithm]['params']
        algorithm_parameters = list(algorithm_params.keys())

        main_list = np.setdiff1d(user_input, algorithm_parameters)

        assert len(main_list) <= 0, "{} are not valid parameters".format(list(main_list))

        for key in list(user_input_params.keys()):

            input_value = user_input_params[key]

            expected_datatype = algorithm_params[key]['datatype']

            # None input flag
            if algorithm_params[key]['default'] == None:
                if input_value == None:
                    continue

            # datatype - string
            if expected_datatype == "str":
                values = algorithm_params[key]['value']
                assert input_value in values, "Please select {} from {}".format(key, values)

            # datatype - int or float
            elif expected_datatype == "int" or expected_datatype == "float":

                # datatype - int
                if expected_datatype == "int":
                    assert type(input_value) == int, "{} should be a integer".format(key)

                if expected_datatype == "float":
                    assert type(input_value) == float or type(input_value) == int, "{} should be a integer".format(key)

                compare_type = algorithm_params[key]['compare_type']

                if compare_type == 'condition':
                    algo_param_value = algorithm_params[key]['condition']['value']
                    algo_condition_symbol = algorithm_params[key]['condition']['symbol']

                    if algo_condition_symbol == ">":
                        assert input_value > algo_param_value, f"{key} should be greater than {algo_param_value}"

                    elif algo_condition_symbol == "<":
                        assert input_value < algo_param_value, f"{key} should be less than {algo_param_value}"

                    elif algo_condition_symbol == "<=":
                        assert input_value <= algo_param_value, f"{key} should be less than or equal to {algo_param_value}"

                    elif algo_condition_symbol == ">=":
                        assert input_value >= algo_param_value, f"{key} should be greater than or equal to {algo_param_value}"

                    elif algo_condition_symbol == "<=>":
                        assert (input_value <= algo_param_value or input_value >= algo_param_value), f"{key} should be <=> to {algo_param_value}"

                if compare_type == 'range':
                    assert algorithm_params[key]['range'][0] <= input_value <= algorithm_params[key]['range'][1], "{} should be in range {}".format(key, algorithm_params[key]['range'])

            # hybrid datatype
            elif expected_datatype == "hybrid":
                algo_param_value = algorithm_params[key]['condition']['value']
                algo_condition_symbol = algorithm_params[key]['condition']['symbol']
                range_lower = algorithm_params[key]['range'][0]
                range_upper = algorithm_params[key]['range'][1]

                if type(input_value) == int or type(input_value) == float:
                    if algo_condition_symbol == ">":
                        condition_flag = input_value > algo_param_value
                    elif algo_condition_symbol == ">=":
                        condition_flag = input_value >= algo_param_value
                    elif algo_condition_symbol == "<=>":
                        condition_flag = input_value <= algo_param_value or input_value >= algo_param_value, "{} should be <=> to {}".format(key, algo_param_value)

                    assert (condition_flag or range_lower <= input_value <= range_upper), "{} should be in range {} or {} to {}".format(key, algorithm_params[key]['range'], algo_condition_symbol, algo_param_value)

                else:
                    print(key)

    except AssertionError as msg:
        print(msg)
        return False
    else:
        return True
