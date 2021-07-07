import json
import os
from dltk_ai.ml_hyperparameters import hyperparameter_dictionary

def get_ml_model_info(info_about, library=None, task=None, algorithm=None):
    """
    This function is to provide list of supported libraries, algorithms & valid hyperparameters
    Args:
        info_about: one of the option from ['libraries', 'algorithms', 'hyperparameters'] about which info is being seeked
        library: ['scikit', 'weka', 'h2o']
        task: ['classification', 'regression']
        algorithm: algorithm name for which more info is required

    Returns:

    """
    try:
        if info_about == 'libraries':
            print(f"Supported libraries for Machine Learning: {list(hyperparameter_dictionary.keys())}")

        elif info_about == 'algorithms':
            assert library in hyperparameter_dictionary, f"Please provide 'library' name out of {list(hyperparameter_dictionary.keys())}"
            assert task in hyperparameter_dictionary[library], f"Please provide 'task' name out of {list(hyperparameter_dictionary[library].keys())}"
            print(f"Supported algorithms for {task.upper()} task in {library.upper()} library are:")
            for algo_name, _ in hyperparameter_dictionary[library][task].items():
                print(f"\n algorithm : {algo_name},\n reference link: {hyperparameter_dictionary[library][task][algo_name]['reference_link']}")

        elif info_about == "hyperparameters":
            assert library in hyperparameter_dictionary, f"Please provide 'library' name out of {list(hyperparameter_dictionary.keys())}"
            assert task in hyperparameter_dictionary[library], f"Please provide 'task' name out of {list(hyperparameter_dictionary[library].keys())}"
            assert algorithm in hyperparameter_dictionary[library][task], f"Please provide 'algorithm' name out of {list(hyperparameter_dictionary[library][task].keys())}"
            supported_hyperparameters_info = hyperparameter_dictionary[library][task][algorithm]['params']

            print(f"Supported hyperparameters for {algorithm.upper()} algorithm in {library.upper()} library are: \n ")
            print('%-55s%-30s' % ("Parameter", "Default Value"))
            for param_name, param_info in supported_hyperparameters_info.items():
                print('%-55s%-30s' % (param_name, param_info['default']))

            print(f"\nfor more info please refer {hyperparameter_dictionary[library][task][algorithm]['reference_link']}")

    except AssertionError as msg:
        print("Error :", msg)
