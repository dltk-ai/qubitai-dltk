import unittest
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from dltk_ai.assertions import hyper_parameter_check


class TestWekaAlgo(unittest.TestCase):

    def setUp(self):
        self.library = "weka"
        pass

    # ------------- LOGISTIC --------------------

    def test_logistic_1(self):
        algorithm = "Logistic"
        service = "classification"
        params = {'-D': True, '-S': False, '-R': True, '-M': 2}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- MULTILAYER PERCEPTRON --------------------

    def test_multilayer_perceptron_1(self):
        algorithm = "MultilayerPerceptron"
        service = "classification"
        params = {'-L': 0.4,'-M': 0.5,'-N': 200,'-V': 50,'-S': 1,'-E': 21,'-A': False,'-B': False,'-H': "a",'-C': False,'-I': False,'-R': False,'-D': False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- NAIVE BAYES MULTINOMIAL --------------------

    def test_naive_bayes_multinomial_1(self):
        algorithm = "NaiveBayesMultinomial"
        service = "classification"
        params = {'-output-debug-info': True,'-do-not-check-capabilities': False,'-num-decimal-places': 3,'-batch-size': 50}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- RANDOM FOREST --------------------

    def test_random_forest_1(self):
        algorithm = "RandomForest"
        service = "classification"
        params = {'-P': 99, '-O': False, '-store-out-of-bag-predictions': False,'-output-out-of-bag-complexity-statistics': False,'-print':False,'-attribute-importance':False,'-I':80,'-num-slots':1,'-K':0,'-M':2,'-V':0.1,'-S':1,'-depth':1,'-N':0,'-U':True,'-B':True,'-output-debug-info':True,'-do-not-check-capabilities':False,'-num-decimal-places':1}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- LibSVM --------------------

    def test_libsvm_1(self):
        algorithm = "LibSVM"
        service = "classification"
        params = {'-S': 3,'-K': 1,'-D': 1,'-G': "1/k",'-R':0,'-C':0.5,'-N':1,'-Z':True,'-J':True,'-V':True,'-P':0.5,'-M':20,'-E':0.1,'-H':False,'-W':1,'-B':False,'-seed':1}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- ADABOOSTM1 --------------------

    def test_adaboostm1_1(self):
        algorithm = "AdaBoostM1"
        service = "classification"
        params = {'-P': 99, '-Q': False, '-S': 2,'-I': 20,'-D':False,'-W':'weka.classifiers.trees.DecisionStump'}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- ATTRIBUTE SELECTED CLASSIFIER --------------------

    def test_attribute_selected_classifier_1(self):
        algorithm = "AttributeSelectedClassifier"
        service = "classification"
        params = {'-E': 'weka.attributeSelection.CfsSubsetEval', '-S': True, '-D': False,'-W': 'weka.classifiers.trees.J48','-U':False,'-C':0.20,'-M':1,'-R':False,'-N':2,'-B':False,'-L':False,'-A':False,'-Q':1}
        if hyper_parameter_check(self.library, service, algorithm, params):
            self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))
        else:
            self.assertFalse(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- BAGGING --------------------

    def test_bagging_1(self):
        algorithm = "Bagging"
        service = "classification"
        params = {'-P': 50, '-O': True, '-print': False,
                  '-store-out-of-bag-predictions': False, '-output-out-of-bag-complexity-statistics': True, '-represent-copies-using-weights': True, '-S': 2, '-num-slots': 1, '-I': 2,
                  '-D': False, '-W': 'weka.classifiers.trees.REPTree', '-M': 1, '-V': 0.1,'-N':1,'-L':-1,'-R':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- KSTAR --------------------

    def test_kstar_1(self):
        algorithm = "KStar"
        service = "classification"
        params = {'-B': 50, '-E': True, '-M': 'a'}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- DECISION TABLE --------------------

    def test_decision_table_1(self):
        algorithm = "DecisionTable"
        service = "classification"
        params = {'-S': 'weka.attributeSelection.BestFirst', '-X': 2, '-E': 'mae','-I':True,'-R':True,'-P':False,'-D':0,'-N':3}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- IBK --------------------

    def test_ibk_1(self):
        algorithm = "IBk"
        service = "classification"
        params = {'-I': False, '-F': True, '-K': 2,'-E':True,'-W':True,'-X':False,'-A':'weka.core.neighboursearch.LinearNNSearch'}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- RANDOM TREE --------------------

    def test_random_tree_1(self):
        algorithm = "RandomTree"
        service = "classification"
        params = {'-K': 0, '-M': 2, '-V': 0.1,'-S':2,'-depth':1,'-N':1,'-U':True,'-B':True,'-output-debug-info':False,'-do-not-check-capabilities':True,'-num-decimal-places':1}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- SMO --------------------

    def test_smo_1(self):
        algorithm = "SMO"
        service = "classification"
        params = {'-no-checks': True, '-C': 2, '-N': 1, '-L': 0.1, '-P': 0.001, '-M': False, '-V': 1, '-W': 1,
                  '-K': 'weka.classifiers.functions.supportVector.PolyKernel', '-claibrator': 'weka.classifiers.functions.Logistic', '-output-debug-info': False,'-do-not-check-capabilities':True,'-num-decimal-places':2}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- LINEAR REGRESSION --------------------

    def test_linear_regression_1(self):
        algorithm = "LinearRegression"
        service = "regression"
        params = {'-S': 2, '-C': True, '-R': 0.00001, '-minimal': False, '-additional-stats': True, '-output-debug-info': False, '-do-not-check-capabilities': False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- ADDITIVE REGRESSION --------------------

    def test_additive_regression_1(self):
        algorithm = "AdditiveRegression"
        service = "regression"
        params = {'-S': 0.4, '-I': 5, '-A': False, '-D': False, '-W': 'weka.classifiers.trees.DecisionStump'}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


if __name__ == '__main__':
    unittest.main()
