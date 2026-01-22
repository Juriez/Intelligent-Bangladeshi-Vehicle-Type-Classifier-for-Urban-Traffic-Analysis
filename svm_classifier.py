import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluate import extract_features  

def train_svm_on_features(train_features, train_labels, kernel='linear', C=1.0):
    """
    Train SVM classifier on extracted features.
    - kernel: 'linear' (fast) or 'rbf' (non-linear, slower but potentially better)
    - C: Regularization parameter (higher = less regularization)
    """
    svm_model = SVC(kernel=kernel, C=C, random_state=42, probability=True)  
    svm_model.fit(train_features, train_labels)
    print(f"SVM trained with kernel='{kernel}', C={C}")
    return svm_model

def evaluate_svm(svm_model, test_features, test_labels, classes, model_name='SVM'):
    y_pred = svm_model.predict(test_features)
    acc = accuracy_score(test_labels, y_pred)
    print(f"\n{model_name} Test Accuracy: {acc*100:.2f}%")
    report = classification_report(test_labels, y_pred, target_names=classes, output_dict=True, digits=2)
    print(classification_report(test_labels, y_pred, target_names=classes, digits=2))

    # Confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return report, acc

def compare_classifiers(original_report, original_acc, svm_report, svm_acc, classes):
    print("\n" + "="*60)
    print("       COMPARISON: Original CNN Classifier vs. SVM")
    print("="*60)
    print(f"Original Accuracy : {original_acc*100:5.2f}%")
    print(f"SVM Accuracy      : {svm_acc*100:5.2f}%")
    print(f"Difference        : {(original_acc - svm_acc)*100:+6.2f}%")
    print("-"*60)

    data = []
    for cls in classes:
        orig_f1 = original_report.get(cls, {}).get('f1-score', 0.0)
        svm_f1 = svm_report.get(cls, {}).get('f1-score', 0.0)
        data.append([cls, f"{orig_f1:.3f}", f"{svm_f1:.3f}", f"{orig_f1 - svm_f1:+.3f}"])

    df = pd.DataFrame(data, columns=['Class', 'Original F1', 'SVM F1', 'Δ (Original - SVM)'])
    print(df.to_string(index=False))
    print("-"*60)
    print(f"Macro Avg F1 → Original: {original_report['macro avg']['f1-score']:.3f} | SVM: {svm_report['macro avg']['f1-score']:.3f}")
    print("="*60 + "\n")