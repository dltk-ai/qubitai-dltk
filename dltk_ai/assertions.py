import re

supported_algorithm = {'regression':

                           {'scikit': ['LinearRegression', 'DecisionTrees', 'Bagging', 'RandomForest',
                                       'GradientBoostingMachines', 'XGradientBoosting', 'AdaBoost', 'ExtraTrees',
                                       'SupportVectorMachines'],
                            'h2o': ['LinearRegression', 'GradientBoostingMachines', 'RandomForest',
                                    'GradientBoostingMachine'],
                            'weka': ['LinearRegression', 'RandomForest', 'AdditiveRegression']
                            },

                       'classification': {
                           'scikit': ['NaiveBayesMultinomial', 'LogisticRegression', 'DecisionTrees', 'Bagging',
                                      'RandomForest', 'GradientBoostingMachines', 'XGradientBoosting', 'AdaBoost',
                                      'ExtraTrees', 'SupportVectorMachines', 'KNearestNeighbour'],
                           'h2o': ['NaiveBayesBinomial', 'DeepLearning'],
                           'weka': ['LibSVM', 'NaiveBayesMultinomial', 'KStar', 'AdaBoostM1',
                                    'AttributeSelectedClassifier', 'Bagging', 'DecisionTable', 'RandomTree', 'SMO',
                                    'Logistic', 'MultilayerPerceptron']
                       },

                       'clustering': {
                           'scikit': ['AgglomerativeClustering', 'KMeansClustering', 'MiniBatchKMeans', 'Birch',
                                      'DBScan', 'SpectralClustering'],
                           'weka': ['SimpleKMeans']
                       }
                       }


def validate_parameters(service: str, library: str, algorithm: str, features: list, label: str,
                        train_percentage: int = 0.80,
                        save_model=True, cluster=False, predict=False):
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

    # if library is scikit or h2o, convert save_model to string
    if (library == 'scikit' or library == 'h2o') and save_model == True:
        save_model = 'true'

    if not save_model:
        print("NOTE: we wont save model, so you can't do prediction")

    return service, library, algorithm, features, label, train_percentage, save_model


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
