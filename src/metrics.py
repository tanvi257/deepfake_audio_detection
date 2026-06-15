import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve

def compute_eer(y_true, y_prob):
    """
    Computes the Equal Error Rate (EER) from true labels and predicted probabilities of the positive class.
    y_true: binary labels (0 for Genuine, 1 for Deepfake)
    y_prob: probability of being Deepfake
    """
    # y_true should be binary (0 and 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    
    # EER is the point where fpr == fnr
    # We find the threshold where absolute difference |fpr - fnr| is minimized
    idx = np.nanargmin(np.absolute(fpr - fnr))
    
    # Average the two rates at this index
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Computes all primary and secondary evaluation metrics.
    """
    overall_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    # cm format: [[TN, FP], [FN, TP]]
    # TN: Genuine classified as Genuine
    # FP: Genuine classified as Deepfake
    # FN: Deepfake classified as Genuine
    # TP: Deepfake classified as Deepfake
    tn, fp, fn, tp = cm.ravel()
    
    genuine_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    deepfake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Equal Error Rate
    eer, eer_threshold = compute_eer(y_true, y_prob)
    
    return {
        'accuracy': overall_acc,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'genuine_accuracy': genuine_acc,
        'deepfake_accuracy': deepfake_acc,
        'eer': eer,
        'eer_threshold': eer_threshold
    }
