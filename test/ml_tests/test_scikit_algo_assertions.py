import unittest
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from dltk_ai.assertions import hyper_parameter_check


class TestScikitClassificationAlgo(unittest.TestCase):

    def setUp(self):
        self.library = "scikit"
        self.service = "classification"
        pass

    # ------------- DECISION TREE --------------------

    def test_decision_tree_1(self):

        # default params

        algorithm = "DecisionTree"

        params = {'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': None,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'splitter': 'best'}

        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_2(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': 0.5,'criterion': 'gun'}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_3(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': 0.3}
        # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_4(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': 0.3}
        # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_randomforest_1(self):
        algorithm = "RandomForest"
        params = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None,
                  'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
                  'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                  'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False,
                  'random_state': None, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_1(self):
        algorithm = "Bagging"
        params = {'base_estimator': None, 'bootstrap': True, 'bootstrap_features': False, 'max_features': 1.0,
                  'max_samples': 1.0, 'n_estimators': 10, 'n_jobs': None, 'oob_score': False, 'random_state': None,
                  'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_1(self):
        algorithm = "ExtraTrees"
        params = {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None,
                  'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0,
                  'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                  'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False,
                  'random_state': None, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_knearestneighbors_1(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None,
                  'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_1(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'SAMME.R', 'base_estimator': None, 'learning_rate': 1.0, 'n_estimators': 50,
                  'random_state': None}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_naivebayesmultinomial_1(self):
        algorithm = "NaiveBayesMultinomial"
        params = {'alpha': 1.0, 'class_prior': None, 'fit_prior': True}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gradientboostingmachines_1(self):
        algorithm = "GradientBoostingMachines"
        params = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'deviance',
                  'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0,
                  'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                  'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': None,
                  'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_1(self):
        algorithm = "XGradientBoosting"
        params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bytree': 1, 'gamma': 0,
                  'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': None,
                  'n_estimators': 100, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0,
                  'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': True, 'subsample': 1}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_1(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0,
                  'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_1(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))


class TestScikitRegressionAlgo(unittest.TestCase):

    def setUp(self):
        self.library = "scikit"
        self.service = "regression"
        pass

    def test_decisiontree_1(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None,
                  'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
                  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': None, 'splitter': 'best'}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_randomforest_1(self):
        algorithm = "RandomForest"
        params = {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto',
                  'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
                  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100,
                  'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_1(self):
        algorithm = "Bagging"
        params = {'base_estimator': None, 'bootstrap': True, 'bootstrap_features': False, 'max_features': 1.0,
                  'max_samples': 1.0, 'n_estimators': 10, 'n_jobs': None, 'oob_score': False, 'random_state': None,
                  'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gradientboostingmachine_1(self):
        algorithm = "GradientBoostingMachine"
        params = {'alpha': 0.9, 'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1,
                  'loss': 'ls', 'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None,
                  'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1,
                  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100,
                  'n_iter_no_change': None, 'random_state': None, 'subsample': 1.0, 'tol': 0.0001,
                  'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_1(self):
        algorithm = "ExtraTrees"
        params = {'bootstrap': False, 'ccp_alpha': 0.0, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto',
                  'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
                  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100,
                  'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_1(self):
        algorithm = "AdaBoost"
        params = {'base_estimator': None, 'learning_rate': 1.0, 'loss': 'linear', 'n_estimators': 50,
                  'random_state': None}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_1(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'cache_size': 200, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.1, 'gamma': 'scale',
                  'kernel': 'rbf', 'max_iter': -1, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))


if __name__ == '__main__':
    unittest.main()
