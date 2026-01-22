import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd

def load_and_evaluate(model, model_path, test_loader, classes, model_name='Model', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} Test Accuracy: {acc*100:.2f}%")
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=2)
    print(classification_report(y_true, y_pred, target_names=classes, digits=2))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return report, acc, y_true, y_pred


def compare_models(resnet_report, resnet_acc, custom_report, custom_acc, classes):
    print("\n" + "="*50)
    print("          MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"ResNet18 Accuracy : {resnet_acc*100:5.2f}%")
    print(f"Custom CNN Accuracy: {custom_acc*100:5.2f}%")
    print(f"Difference         : {(resnet_acc - custom_acc)*100:+6.2f}%")
    print("-"*50)

    comparison_data = []
    for cls in classes:
        r_f1 = resnet_report.get(cls, {}).get('f1-score', 0.0)
        c_f1 = custom_report.get(cls, {}).get('f1-score', 0.0)
        comparison_data.append([cls, f"{r_f1:.3f}", f"{c_f1:.3f}", f"{r_f1 - c_f1:+.3f}"])

    df = pd.DataFrame(comparison_data, columns=['Class', 'ResNet F1', 'Custom F1', 'Δ (ResNet - Custom)'])
    print(df.to_string(index=False))

    print("-"*50)
    print(f"Macro Avg F1 → ResNet: {resnet_report['macro avg']['f1-score']:.3f} | Custom: {custom_report['macro avg']['f1-score']:.3f}")
    print("="*50 + "\n")
    
    
def extract_features(model, loader, model_type='resnet', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            
            if model_type == 'resnet':
                
                feat = model.avgpool(
                    model.layer4(
                        model.layer3(
                            model.layer2(
                                model.layer1(
                                    model.bn1(
                                        model.conv1(images)
                                    )
                                )
                            )
                        )
                    )
                )
                
                feat = feat.view(feat.size(0), -1) 
                
            elif model_type == 'custom_cnn':
               
                feat = model.pool(model.relu(model.conv3(
                    model.pool(model.relu(model.conv2(
                        model.pool(model.relu(model.conv1(images)))
                    )))
                )))
                feat = feat.view(feat.size(0), -1)
                
            else:
                raise ValueError("Unsupported model_type")
            
            features_list.append(feat.cpu().numpy())
            labels_list.append(lbls.numpy())
    
    
    features = np.concatenate(features_list, axis=0)
    labels   = np.concatenate(labels_list, axis=0)
    
    print(f"Extracted {features.shape[0]} samples with feature dim {features.shape[1]}")
    return features, labels