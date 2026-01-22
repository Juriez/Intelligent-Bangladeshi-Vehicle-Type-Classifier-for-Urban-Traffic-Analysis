from preprocess import get_data_loaders, classes, num_classes
from model import get_resnet18, get_custom_cnn
from evaluate import load_and_evaluate, extract_features
from svm_classifier import train_svm_on_features, evaluate_svm, compare_classifiers
import torch
import numpy as np

if __name__ == "__main__":
    print("=== Starting full classification systems comparison ===")
    print("Loading data loaders (train, val, test)...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")


    print("Loading saved ResNet18 model for evaluation...")
    resnet_model = get_resnet18(num_classes)
    try:
        resnet_report, resnet_acc, _, _ = load_and_evaluate(
            model=resnet_model,
            model_path='vehicle_classifier.pth',           
            test_loader=test_loader,
            classes=classes,
            model_name='ResNet18 (Softmax/CE)',
            device=device
        )
    except FileNotFoundError:
        print("Error: 'resnet_classifier.pth' not found! Update the model_path.")
        exit(1)

 
    print("\nLoading saved Custom CNN model for evaluation...")
    custom_model = get_custom_cnn(num_classes)
    try:
        custom_report, custom_acc, _, _ = load_and_evaluate(
            model=custom_model,
            model_path='custom_cnn_classifier.pth',       
            test_loader=test_loader,
            classes=classes,
            model_name='Custom CNN (Softmax/CE)',
            device=device
        )
    except FileNotFoundError:
        print("Error: 'custom_cnn_classifier.pth' not found! Update the model_path.")
        exit(1)

   
    print("\n" + "="*70)
    print("Extracting features using ResNet18...")
    train_features_resnet, train_labels = extract_features(
        resnet_model, train_loader, model_type='resnet', device=device
    )
    test_features_resnet, test_labels = extract_features(
        resnet_model, test_loader, model_type='resnet', device=device
    )
    print(f"ResNet features: Train shape = {train_features_resnet.shape}, Test shape = {test_features_resnet.shape}")

    print("\nExtracting features using Custom CNN...")
    train_features_custom, _ = extract_features(
        custom_model, train_loader, model_type='custom_cnn', device=device
    )
    test_features_custom, _ = extract_features(
        custom_model, test_loader, model_type='custom_cnn', device=device
    )
    print(f"Custom CNN features: Train shape = {train_features_custom.shape}, Test shape = {test_features_custom.shape}")
    
    # ADD FEATURE NORMALIZATION 
    from sklearn.preprocessing import StandardScaler

    print("\nNormalizing features with StandardScaler...")

    # For ResNet features
    scaler_resnet = StandardScaler()
    train_features_resnet = scaler_resnet.fit_transform(train_features_resnet)
    test_features_resnet  = scaler_resnet.transform(test_features_resnet)

    # For Custom CNN features
    scaler_custom = StandardScaler()
    train_features_custom = scaler_custom.fit_transform(train_features_custom)
    test_features_custom  = scaler_custom.transform(test_features_custom)

    print("Feature normalization completed.")

    print("\n" + "="*70)
    print("Training & evaluating SVM using ResNet18 features...")
    svm_resnet = train_svm_on_features(
        train_features_resnet, train_labels, kernel='linear', C=1.0
    )
    svm_resnet_report, svm_resnet_acc = evaluate_svm(
        svm_resnet, test_features_resnet, test_labels, classes,
        model_name='SVM (ResNet18 features)'
    )
    
    print("\n" + "="*70)
    print("Training & evaluating SVM using Custom CNN features...")
    svm_custom = train_svm_on_features(
        train_features_custom, train_labels, kernel='linear', C=1.0
    )
    svm_custom_report, svm_custom_acc = evaluate_svm(
        svm_custom, test_features_custom, test_labels, classes,
        model_name='SVM (Custom CNN features)'
    )

    print("\n" + "="*80)
    print("                  FINAL COMPARISONS")
    print("="*80)

    print("\nComparison 1: ResNet18 (Softmax) vs SVM (ResNet features)")
    compare_classifiers(
        resnet_report, resnet_acc,
        svm_resnet_report, svm_resnet_acc,
        classes
    )

    print("\nComparison 2: Custom CNN (Softmax) vs SVM (Custom CNN features)")
    compare_classifiers(
        custom_report, custom_acc,
        svm_custom_report, svm_custom_acc,
        classes
    )

    print("\nAll evaluations complete.")
    print("You can now compare how SVM performs with strong (ResNet) vs weaker (Custom CNN) features.")