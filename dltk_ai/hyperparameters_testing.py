from dltk_ai.assertions import validate_parameters, is_url_valid, allowed_file_extension, hyper_parameter_check

def params_check(service, algorithm, label, features, lib="weka", train_percentage=80, params=None):

    service, library, algorithm, features, label, train_percentage, save_model = validate_parameters(service, lib, algorithm, features, label, train_percentage)

    hyper_parameter_check(library, service, algorithm, params)

    return params