import json
import os

script_dir = os.path.dirname(__file__)

with open(os.path.join(script_dir, "./ml_hyperparameters.json"), 'r') as file:
    model_info = json.load(file)


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
            print(f"Supported libraries for Machine Learning: {list(model_info.keys())}")

        elif info_about == 'algorithms':
            assert library in model_info, f"Please provide 'library' name out of {list(model_info.keys())}"
            assert task in model_info[library], f"Please provide 'task' name out of {list(model_info[library].keys())}"
            print(f"Supported algorithms for {task.upper()} task in {library.upper()} library are:")
            for algo_name, _ in model_info[library][task].items():
                print(f"\n algorithm : {algo_name},\n 'reference link': {model_info[library][task][algo_name]['reference_link']}")

        elif info_about == "hyperparameters":
            assert library in model_info, f"Please provide 'library' name out of {list(model_info.keys())}"
            assert task in model_info[library], f"Please provide 'task' name out of {list(model_info[library].keys())}"
            assert algorithm in model_info[library][task], f"Please provide 'algorithm' name out of {list(model_info[library][task].keys())}"
            supported_hyperparameters_info = model_info[library][task][algorithm]['params']

            print(f"Supported hyperparameters for {algorithm.upper()} algorithm in {library.upper()} library are: \n ")
            print('%-55s%-30s%-6s' % ("Parameter", "Default Value", "Data Type"))
            for param_name, param_info in supported_hyperparameters_info.items():
                print('%-55s%-30s%-6s' % (param_name, param_info['default'], param_info['datatype']))

            print(f"\nfor more info please refer {model_info[library][task][algorithm]['reference_link']}")

    except AssertionError as msg:
        print("Error :", msg)
