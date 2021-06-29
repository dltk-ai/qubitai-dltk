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
        params = {'ccp_alpha': 0.0,'class_weight': None,'criterion':'gini','max_depth': None,'max_features': None, 'max_leaf_nodes': None,'min_impurity_decrease': 0.0,'min_samples_leaf': 1,'min_samples_split': 2,'min_weight_fraction_leaf': 0.0,'splitter': 'best'}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_2(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': 0.5,'criterion': 'gun'}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_3(self):
        algorithm = "DecisionTree"
        params = {'max_depth': 0.5,'max_features': 'gun','max_leaf_nodes': 1.0}
        # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_4(self):
        algorithm = "DecisionTree"
        params = {'max_depth': 5,'max_features': 'gun','max_leaf_nodes': 2}
        # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_5(self):
        algorithm = "DecisionTree"
        params = {'min_impurity_decrease': -0.5,'min_samples_leaf': 'gun','min_samples_split': 1}
        # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_6(self):
        algorithm = "DecisionTree"
        params = {'min_impurity_decrease': 0.5,'min_samples_leaf': 0.7,'min_samples_split': 1}
        # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_7(self):
        algorithm = "DecisionTree"
        params = {'min_impurity_decrease': 0.5,'min_samples_leaf': 0.3,'min_samples_split': 1}
        # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_8(self):
        algorithm = "DecisionTree"
        params = {'min_weight_fraction_leaf': 0.5,'splitter': 'best'}
        # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    # ------------- RANDOM FOREST -------------------- #

    def test_random_forest_1(self):
        algorithm = "RandomForest"
        params = {'bootstrap': True, 'ccp_alpha': 0.0,'class_weight': None,'criterion': 'gini','max_depth': None,'max_features': 'auto','max_leaf_nodes': None,'max_samples': None,'min_impurity_decrease': 0.0,'min_impurity_split': None,'min_samples_leaf': 1,'min_samples_split': 2,'min_weight_fraction_leaf': 0.0,'n_estimators': 100,'n_jobs': None,'oob_score': False,'verbose': 0,'warm_start': False}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_2(self):
        algorithm = "RandomForest"
        params = {'bootstrap': "true", 'ccp_alpha': -2}#,'class_weight': None,'criterion': 'gini','max_depth': None,'max_features': 'auto','max_leaf_nodes': None,'max_samples': None,'min_impurity_decrease': 0.0,'min_impurity_split': None,'min_samples_leaf': 1,'min_samples_split': 2,'min_weight_fraction_leaf': 0.0,'n_estimators': 100,'n_jobs': None,'oob_score': False,'verbose': 0,'warm_start': False}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_3(self):
        algorithm = "RandomForest"
        params = {'class_weight': "random_value_cause_except",'criterion': 'gini','max_depth': 0.34}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_4(self):
        algorithm = "RandomForest"
        params = {'class_weight': "random_value_cause_except",'criterion': 'gini','max_depth': 3}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_5(self):
        algorithm = "RandomForest"
        params = {'max_features': 'random_value_cause_except','max_leaf_nodes': 1.5}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_6(self):
        algorithm = "RandomForest"
        params = {'max_features': 'random_value_cause_except','max_leaf_nodes': 3}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_7(self):
        algorithm = "RandomForest"
        params = {'min_impurity_decrease': 0.0,'min_impurity_split': None}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_8(self):
        algorithm = "RandomForest"
        params = {'min_impurity_decrease': 0.0,'min_impurity_split': -0.4}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_9(self):
        algorithm = "RandomForest"
        params = {'min_samples_leaf': -0.7,'min_samples_split': 0.5}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_10(self):
        algorithm = "RandomForest"
        params = {'n_estimators': -100}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_11(self):
        algorithm = "RandomForest"
        params = {'min_weight_fraction_leaf': 1,'n_estimators': 0}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_random_forest_12(self):
        algorithm = "RandomForest"
        params = {'min_weight_fraction_leaf': 0.2,'n_estimators': 10}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    # ------------- BAGGING -------------------- #

    def test_bagging_1(self):
        algorithm = "Bagging"
        params = {'base_estimator': None, 'bootstrap': True, 'bootstrap_features': False, 'max_features': 1.0,'max_samples': 1.0,'n_estimators': 10,'n_jobs': None,'oob_score': False,'random_state': None,'verbose': 0, 'warm_start': False}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_2(self):
        algorithm = "Bagging"
        params = {'bootstrap': "false"}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_3(self):
        algorithm = "Bagging"
        params = {'bootstrap': False, 'bootstrap_features':True}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_4(self):
        algorithm = "Bagging"
        params = {'bootstrap': False, 'bootstrap_features':True, 'max_features': -30}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_5(self):
        algorithm = "Bagging"
        params = {'bootstrap': False, 'bootstrap_features':True, 'max_features': 30, 'max_samples': "can_be_anything"}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_6(self):
        algorithm = "Bagging"
        params = {'bootstrap': False, 'bootstrap_features':True, 'max_features': 30, 'max_samples': "can_be_anything",'n_estimators':-100}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_7(self):
        algorithm = "Bagging"
        params = {'bootstrap': False, 'bootstrap_features':True, 'max_features': 30, 'max_samples': "can_be_anything",'n_estimators':100,'oob_score':"fas"}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_bagging_8(self):
        algorithm = "Bagging"
        params = {'bootstrap': False, 'bootstrap_features':True, 'max_features': 30, 'max_samples': "can_be_anything",'n_estimators':100,'oob_score':False}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    # ------------- ExtraTrees -------------------- #

    def test_extratrees_1(self):
        algorithm = "ExtraTrees"
        params = {'bootstrap': False,'ccp_alpha': 0.0,'class_weight': None,'criterion': 'gini','max_depth': None,'max_features': 'auto','max_leaf_nodes': None,'max_samples': None,'min_impurity_decrease': 0.0,'min_impurity_split':None,'min_samples_leaf': 1,'min_samples_split': 2,'min_weight_fraction_leaf': 0.0,'n_estimators': 100,'n_jobs': None,'oob_score': False,'random_state': None,'verbose': 0,'warm_start': False}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_2(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': -0.1}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_3(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'ginient'}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_4(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything'}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_5(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':1}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))


    def test_extratrees_6(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':2, "max_samples":"canbeanything"}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_7(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes': 2,
     "max_samples": "canbeanything", "min_impurity_decrease": 0}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_8(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes': 2,
     "max_samples": "canbeanything", "min_impurity_decrease": 1}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_9(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes': 2,
     "max_samples": "canbeanything", "min_samples_leaf": 0.7}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_10(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':2, "max_samples":"canbeanything", "min_samples_split":0.2}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_11(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':2, "max_samples":"canbeanything", "min_weight_fraction_leaf":-0.3}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_12(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':2, "max_samples":"canbeanything", "min_samples_leaf":0.2}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_13(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':2, "max_samples":"canbeanything", "n_estimators":0.4}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_extratrees_14(self):
        algorithm = "ExtraTrees"
        params = {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_features': 'canbeanything', 'max_leaf_nodes':2, "max_samples":"canbeanything", "min_samples_leaf":1,"n_estimators":300,"min_weight_fraction_leaf":0.3}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    # ------------- KNN -------------------- #

    def test_knn_1(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_knn_2(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'randomname'}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_knn_3(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'kd_tree', 'leaf_size': -30}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_knn_4(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric': 'minkowski'}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_knn_5(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric':"chebyshev", 'metric_params': None,'n_neighbors': -5.0}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_knn_6(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric':"chebyshev", 'metric_params': None,'n_neighbors': 5, 'p':1}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))


    def test_knn_7(self):
        algorithm = "KNearestNeighbors"
        params = {'algorithm': 'ball_tree', 'leaf_size': 30, 'metric':"chebyshev", 'metric_params': None,'n_neighbors': 5, 'p':3}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    # ------------- AdaBoost -------------------- #

    def test_adaboost_1(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'SAMME.R','base_estimator': None,'learning_rate': 1.0,'n_estimators': 50,'random_state': None}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_2(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'afaSAMME.R'}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_3(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'SAMME.R','learning_rate': -1.0}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_4(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'SAMME.R','learning_rate': 1.0, 'n_estimators': 5.2}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_5(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'SAMME.R','learning_rate': 1.0, 'n_estimators': 54}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_adaboost_6(self):
        algorithm = "AdaBoost"
        params = {'algorithm': 'SAMME','learning_rate': 1.0, 'n_estimators': 54}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    # ------------- NaiveBayesMultinomial -------------------- #

    def test_naivebayes_1(self):
        algorithm = "NaiveBayesMultinomial"
        params = {'alpha': 1.0, 'class_prior': None, 'fit_prior': True}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_naivebayes_2(self):
        algorithm = "NaiveBayesMultinomial"
        params = {'alpha': -1.0, 'class_prior': None, 'fit_prior': True}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_naivebayes_3(self):
        algorithm = "NaiveBayesMultinomial"
        params = {'alpha': 5.6, 'class_prior': None, 'fit_prior': False}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_naivebayes_4(self):
        algorithm = "NaiveBayesMultinomial"
        params = {'alpha': 0, 'class_prior': None, 'fit_prior': True}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

# ------------- GradientBoostingMachines -------------------- #

    def test_gbm_1(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 0.0,'criterion': 'friedman_mse','init': None,'learning_rate': 0.1,'loss': 'deviance','max_depth': 3,'max_features': None,'max_leaf_nodes': None,'min_impurity_decrease': 0.0,'min_impurity_split': None,'min_samples_leaf': 1,'min_samples_split': 2,'min_weight_fraction_leaf': 0.0,'n_estimators': 100,'n_iter_no_change': None,'random_state': None,'subsample': 1.0,'tol': 0.0001,'validation_fraction': 0.1,'verbose': 0,'warm_start': False}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_2(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': -1.9}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_3(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'somethingrandom'}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_4(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential'}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_5(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 0}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_6(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 2.3}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_7(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "min_impurity_split":-3}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_8(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "min_impurity_split":-3}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_8(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "min_impurity_split":3, "min_samples_leaf":0.5}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_9(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "min_impurity_split":3, "min_samples_leaf":0.5,"min_samples_split":4}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_10(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "min_impurity_split":3, "min_samples_leaf":0.7,"min_samples_split":0}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_11(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "min_impurity_split":3, "min_samples_leaf":1,"min_samples_split":-10}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_12(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "n_iter_no_change":0}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_13(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "n_iter_no_change":1,"subsample":2.6}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_14(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,'criterion': 'mse', 'loss': 'exponential','max_depth': 2,'max_leaf_nodes': 5, "min_impurity_decrease":2.3, "n_iter_no_change":1,"subsample":0.6}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_15(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,"validation_fraction":30}
       # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_gbm_15(self):
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 1.3,"validation_fraction":0.5}
       # assertion should fail
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

        # ------------- XGradientBoosting -------------------- #


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
        algorithm = "GradientBoostingMachine"
        params = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'deviance',
                  'max_depth': 3, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0,
                  'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                  'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_iter_no_change': None, 'random_state': None,
                  'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

#----------------------XGradientBoosting-------------------------#

    def test_xgradientboosting_1(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': None,'colsample_bynode': None, 'colsample_bytree': None,'gamma': None,'learning_rate': None,'max_delta_step': None,'max_depth': None, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_2(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 2.5,'colsample_bynode': -2.4, 'colsample_bytree': None,'gamma': None,'learning_rate': None,'max_delta_step': None,'max_depth': None, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_3(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 2.5,'colsample_bynode': -2.4, 'colsample_bytree': None,'gamma': -1.2,'learning_rate': None,'max_delta_step': None,'max_depth': None, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_3(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 0.5,'colsample_bynode': 0.4, 'colsample_bytree': None,'gamma': -1.2,'learning_rate': None,'max_delta_step': None,'max_depth': None, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_4(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 0.5,'colsample_bynode': 0.4, 'colsample_bytree': None,'gamma': 10,'learning_rate': -1,'max_delta_step': None,'max_depth': None, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_5(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 0.5,'colsample_bynode': 0.4, 'colsample_bytree': None,'gamma': 10,'learning_rate': 2,'max_delta_step': 0,'max_depth': -1, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_6(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 0.5,'colsample_bynode': 0.4, 'colsample_bytree': None,'gamma': 10,'learning_rate': 2,'max_delta_step': 0,'max_depth': 3, 'min_child_weight': None,'missing': None,'n_estimators': -100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_xgradientboosting_7(self):
        algorithm = "XGradientBoosting"
        params = {'objective': 'binary:logistic','base_score': 123,'booster': None,'colsample_bylevel': 0.5,'colsample_bynode': 0.4, 'colsample_bytree': None,'gamma': 10,'learning_rate': 2,'max_delta_step': 0,'max_depth': 3, 'min_child_weight': None,'missing': None,'n_estimators': 100,'n_jobs': None,'random_state': None,'reg_alpha': None,'reg_lambda': None,'scale_pos_weight': None, 'subsample': None,'tree_method': None}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

#------------------------------------ SupportVectorMachines ----------------------------------------#
    def test_supportvectormachines_1(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0,
                  'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_2(self):
        algorithm = "SupportVectorMachines"
        params = {'C': -1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0,
                  'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_3(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': "canbeanything",
                  'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_4(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': "canbeanything",
                  'decision_function_shape': 'ovr', 'degree': -3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_5(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': "canbeanything",
                  'decision_function_shape': 'ovr', 'degree': 30, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1.9,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_6(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': "canbeanything",
                  'decision_function_shape': 'ovr', 'degree': 30, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -3,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': -0.001, 'verbose': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_supportvectormachines_7(self):
        algorithm = "SupportVectorMachines"
        params = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': "canbeanything",
                  'decision_function_shape': 'ovr', 'degree': 30, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -3,
                  'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

#-----------------------LogisticRegression--------------------------------#

    def test_logisticregression_1(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_2(self):
        algorithm = "LogisticRegression"
        params = {'C': -1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_3(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': -1.90,
                  'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_4(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1.89,
                  'l1_ratio': None, 'max_iter': -10.0, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_5(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': "100", 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_6(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'randomstring',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_7(self):
        algorithm = "LogisticRegression"
        params = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
                  'random_state': None, 'solver': 'lbfgs', 'tol': -0.0001, 'verbose': 0, 'warm_start': False}
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_logisticregression_8(self):
        algorithm = "LogisticRegression"
        params = {'C': 4.0, 'class_weight': None, 'dual': False, 'fit_intercept': False, 'intercept_scaling': 1,
                  'l1_ratio': None, 'max_iter': 10000, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l1',
                  'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))


class TestScikitRegressionAlgo(unittest.TestCase):

    def setUp(self):
        self.library = "scikit"
        self.service = "regression"
        pass

#----------------- DecisionTrees -------------------#
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