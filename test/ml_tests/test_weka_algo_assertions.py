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
        params = { '-S': False, '-M': 2}
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
        params = {'-S': 3,'-K': 1,'-D': 1,'-R':0,'-C':0.5,'-N':1,'-Z':True,'-J':True,'-V':True,'-P':0.5,'-M':20,'-E':0.1,'-H':False,'-W':1,'-B':False,'-seed':1}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- ADABOOSTM1 --------------------

    def test_adaboostm1_1(self):
        algorithm = "AdaBoostM1"
        service = "classification"
        params = {'-P': 99, '-Q': False, '-S': 2,'-I': 20,'-D':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- ATTRIBUTE SELECTED CLASSIFIER --------------------

    def test_attribute_selected_classifier_1(self):
        algorithm = "AttributeSelectedClassifier"
        service = "classification"
        params = {  '-D': False,'-U':False,'-R':False,'-B':False,'-L':False,'-A':False}
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
                  '-D': False, '-R':False}
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
        params = {'-X': 2, '-E': 'mae','-I':True,'-R':True,'-P':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))


    # ------------- IBK --------------------

    def test_ibk_1(self):
        algorithm = "IBk"
        service = "classification"
        params = {'-I': False, '-F': True, '-K': 2,'-E':True,'-W':True,'-X':False,}
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
                    '-output-debug-info': False,'-do-not-check-capabilities':True,'-num-decimal-places':2}
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
        params = {'-S': 0.4, '-I': 5, '-A': False, '-D': False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- HierarchicalClusterer --------------------

    def test_hierarchical_clusterer_1(self):
        algorithm = "HierarchicalClusterer"
        service = "clustering"
        params = {'-L':'SINGLE','-P':False,'-D':False,'-B':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- EM --------------------

    def test_em_1(self):
        algorithm = "EM"
        service = "clustering"
        params = {'-X':5,'-K':10,'-max':-1,'-ll-cv':1e-06,'-I':100,'-ll-iter':1e-06,'-V':False,'-M':1e-06,'-O':False,'-num-slots':1,'-S':100,'-output-debug-info':False,'-do-not-check-capabilities':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- K-MEANS --------------------

    def test_kmeans_1(self):
        algorithm = "SimpleKMeans"
        service = "clustering"
        params = {'-init':0,'-C':False,'-max-candidates':100,'-periodic-pruning':10000,'-min-density':2,'-t2':-1.0,'-t1':-1.5,'-V':False,'-M':False,'-I':1,'-O':False,'-fast':False,'-num-slots':1,'-S':10,'-output-debug-info':False,'-do-not-check-capabilities':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- MakeDensityBasedClusterer --------------------

    def test_make_density_based_clusterer_1(self):
        algorithm = "MakeDensityBasedClusterer"
        service = "clustering"
        params = {'-M':1e-06,'-S':10,'-V':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- FarthestFirst --------------------

    def test_farthest_first_1(self):
        algorithm = "FarthestFirst"
        service = "clustering"
        params = {'-S':1}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))

    # ------------- Canopy --------------------

    def test_canopy_1(self):
        algorithm = "Canopy"
        service = "clustering"
        params = {'-max-candidates':100,'-periodic-pruning':10000,'-min-density':2,'-t2':-1.0,'-t1':-1.5,'-M':False,'-S':1,'-output-debug-info':False,'-do-not-check-capabilities':False}
        self.assertTrue(hyper_parameter_check(self.library, service, algorithm, params))



if __name__ == '__main__':
    unittest.main()
