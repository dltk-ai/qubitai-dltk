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
        algorithm = "DecisionTree"
        params = {'ccp_alpha': 0.5}
        self.assertTrue(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_2(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': "0.5"}
        # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))

    def test_decision_tree_3(self):
        algorithm = "DecisionTree"
        params = {'ccp_alpha': -0.3}
        # assertion should fail
        self.assertFalse(hyper_parameter_check(self.library, self.service, algorithm, params))


if __name__ == '__main__':
    unittest.main()
