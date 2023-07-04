import numpy as np
from sklearn.metrics import confusion_matrix

# Calcul des coûts d'un faux négatif (FN) et d'un faux positif (FP)
cost_fn = 10  # Coût d'un FN
cost_fp = 1   # Coût d'un FP

# Fonction pour calculer le score métier
def calculate_business_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # Matrice de confusion
    fn = cm[1, 0]  # Faux négatifs dans la matrice de confusion
    fp = cm[0, 1]  # Faux positifs dans la matrice de confusion
    business_score = fn * cost_fn + fp * cost_fp  # Calcul du score métier
    return -business_score
