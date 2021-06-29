import unittest
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from dltk_ai.assertions import hyper_parameter_check



class TestH2OClassificationAlgo(unittest.TestCase):

    def setUp(self):
        self.library = "h2o"
        self.service = "regression"
        pass

    def test_linearregression_1(self):
        algorithm = "LinearRegression"
        params = {'HGLM': False, 'auc_type': 'auto', 'balance_classes': False, 'beta_epsilon': 0.0001, 'calc_like': False,
                  'cold_start': False, 'compute_p_values': False, 'early_stopping': True, 'family': 'auto',
                  'fold_assignment': 'auto', 'gradient_epsilon': -1, 'ignore_const_cols': True, 'intercept': True,
                  'keep_cross_validation_fold_assignment': False, 'keep_cross_validation_models': True,
                  'keep_cross_validation_predictions': False, 'lambda_min_ratio': -1, 'lambda_search': False,
                  'link': 'family_default', 'max_active_predictors': -1, 'max_after_balance_size': 5.0,
                  'max_confusion_matrix_size': 20, 'max_iterations': -1, 'max_runtime_secs': 0,
                  'missing_values_handling': 'mean_imputation', 'nlambdas': -1, 'non_negative': False, 'obj_reg': -1,
                  'objective_epsilon': -1, 'prior': -1, 'remove_collinear_columns': False, 'score_each_iteration': False,
                  'score_iteration_interval': -1, 'seed': -1, 'solver': 'auto', 'standardize': True,
                  'stopping_metric': 'auto', 'stopping_rounds': 0, 'stopping_tolerance': 0.001, 'theta': 1e-10,
                  'tweedie_link_power': 1, 'tweedie_variance_power': 0}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gradientboostingmachines_1(self):
        algorithm = "GradientBoostingMachines"
        params = {'auc_type': 'auto', 'balance_classes': False, 'build_tree_one_node': False, 'calibrate_model': False,
                  'categorical_encoding': 'auto', 'check_constant_response': True, 'col_sample_rate': 1,
                  'col_sample_rate_change_per_level': 1, 'col_sample_rate_per_tree': 1, 'distribution': 'auto',
                  'fold_assignment': 'auto', 'gainslift_bins': -1, 'histogram_type': 'auto', 'huber_alpha': 0.9,
                  'ignore_const_cols': True, 'keep_cross_validation_fold_assignment': False,
                  'keep_cross_validation_models': True, 'keep_cross_validation_predictions': False, 'learn_rate': 0.1,
                  'learn_rate_annealing': 1, 'max_abs_leafnode_pred': 1.7976, 'max_after_balance_size': 5.0,
                  'max_confusion_matrix_size': 20, 'max_depth': 5, 'max_runtime_secs': 0, 'min_rows': 10,
                  'min_split_improvement': 1e-05, 'nbins': 20, 'nbins_cats': 1024, 'nbins_top_level': 1024, 'ntrees': 50,
                  'pred_noise_bandwidth': 0, 'quantile_alpha': 0.5, 'r2_stopping': 1.7976, 'sample_rate': 1,
                  'score_each_iteration': False, 'score_tree_interval': 0, 'seed': -1, 'stopping_metric': 'auto',
                  'stopping_rounds': 0, 'stopping_tolerance': 0.001, 'tweedie_power': 1.5}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_randomforest_1(self):
        algorithm = "RandomForest"
        params = {'auc_type': 'auto', 'balance_classes': False, 'binomial_double_trees': False,
                  'build_tree_one_node': False, 'calibrate_model': False, 'categorical_encoding': 'auto',
                  'check_constant_response': True, 'col_sample_rate_change_per_level': 1, 'col_sample_rate_per_tree': 1,
                  'distribution': 'auto', 'fold_assignment': 'auto', 'histogram_type': 'auto',
                  'keep_cross_validation_fold_assignment': False, 'keep_cross_validation_models': True,
                  'keep_cross_validation_predictions': False, 'max_after_balance_size': 1.0, 'max_confusion_matrix_size': 0,
                  'max_depth': 0, 'max_runtime_secs': 0, 'min_rows': 1, 'min_split_improvement': 1e-05, 'mtries': -1,
                  'nbins': 2, 'nbins_cats': 1024, 'nbins_top_level': 1024, 'nfolds': 0, 'ntrees': 50,
                  'r2_stopping': -1.7976, 'sample_rate': 0.632, 'score_each_iteration': False, 'score_tree_interval': 10,
                  'stopping_metric': 'auto', 'stopping_rounds': 10, 'stopping_tolerance': 0.001, 'calibration_frame': None,
                  'checkpoint': None, 'class_sampling_factors': None, 'custom_metric_func': None,
                  'export_checkpoints_dir': None, 'ignore_const_cols': None, 'ignored_columns': None, 'model_id': None,
                  'offset_column': None, 'response_column': None, 'sample_rate_per_class': None, 'seed': None,
                  'weights_column': None}

        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))


if __name__ == '__main__':
    unittest.main()
