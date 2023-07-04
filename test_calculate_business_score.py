import unittest
import pandas as pd
from sklearn.metrics import confusion_matrix
from business_score import calculate_business_score

class TestCalculateBusinessScore(unittest.TestCase):
    def test_calculate_business_score(self):
        # Créer des exemples de dataframe pour le test
        y_true_1 = pd.Series([0, 1, 1, 0])
        y_pred_1 = pd.Series([1, 1, 0, 0])
        y_true_2 = pd.Series([0, 1, 0, 1, 1])
        y_pred_2 = pd.Series([0, 1, 1, 0, 1])

        # Appeler la fonction calculate_business_score pour les différents cas
        score_1 = calculate_business_score(y_true_1, y_pred_1)
        score_2 = calculate_business_score(y_true_2, y_pred_2)

        # Calculer les matrices de confusion pour vérifier les résultats
        cm_1 = confusion_matrix(y_true_1, y_pred_1)
        fn_1 = cm_1[1, 0]
        fp_1 = cm_1[0, 1]
        expected_score_1 = fn_1 * 10 + fp_1 * 1

        cm_2 = confusion_matrix(y_true_2, y_pred_2)
        fn_2 = cm_2[1, 0]
        fp_2 = cm_2[0, 1]
        expected_score_2 = fn_2 * 10 + fp_2 * 1

        # Vérifier si les scores calculés correspondent aux scores attendus
        self.assertEqual(score_1, -expected_score_1)
        self.assertEqual(score_2, -expected_score_2)

if __name__ == '__main__':
    unittest.main()
