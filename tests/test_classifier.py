import unittest
from clasificador import NaiveBayesClassifier

class ClassifierTest(unittest.TestCase):
    def test_predictions_and_long_document_probabilities(self):
        model = NaiveBayesClassifier()
        model.entrenar([[10, 0], [0, 10]], [0, 1])
        self.assertEqual(model.predict([[10000, 0], [0, 10000]]), [0, 1])
        for probability in model.predict_probabilities([[10000, 0], [0, 10000]]):
            self.assertAlmostEqual(sum(probability.values()), 1.0)

    def test_empty_training_is_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().entrenar([], [])

if __name__ == '__main__':
    unittest.main()
