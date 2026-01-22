# Intelligent Bangladeshi Vehicle Type Classifier

A deep learning-based multi-class image classifier for recognizing **10 native Bangladeshi vehicle types** commonly found in Dhaka's chaotic urban traffic. Built using the **Poribohon-BD** dataset, the project compares **transfer learning (ResNet18)** vs **custom CNN**, and **softmax classification** vs **SVM on extracted features**.

<img width="640" height="359" alt="Dhaka Traffic Scene" src="https://github.com/user-attachments/assets/df8a622c-b5fe-4804-812f-dc47a873b370" />

## Features & Highlights

- Classifies 10 Bangladeshi vehicle types: bicycle, bike, bus, car, cng, easybike, leguna, rickshaw, truck, van
- Uses **Poribohon-BD** dataset (~9,058 real-world images)
- Implements and compares:
  - Pre-trained **ResNet18** (transfer learning) → **80% accuracy**
  - Custom lightweight CNN (from scratch) → **69% accuracy**
  - SVM classifier on features extracted from both models
- Handles class imbalance, data augmentation, confusion matrix analysis
- Modular code structure (PyTorch + scikit-learn)

## Project Goals

- Build a culturally relevant vehicle classifier for Dhaka traffic
- Demonstrate power of transfer learning vs training from scratch
- Compare deep learning (softmax) vs classical ML (SVM) on same features
- Provide foundation for future ITS applications in Bangladesh

## Results Summary (Test Set – 518 images)

| Model / Approach                        | Accuracy | Macro F1 | Notes                                      |
|-----------------------------------------|----------|----------|--------------------------------------------|
| ResNet18 + Softmax (transfer learning)  | 80%      | 0.70     | Best overall performance                   |
| Custom CNN + Softmax (from scratch)     | 69%      | 0.59     | Noticeable drop without pre-training       |
| SVM on ResNet18 features                | 55.02%   | 0.50     | Linear kernel underperformed vs softmax    |
| SVM on Custom CNN features              | ~55–60%  | ~0.50    | Weak features → poor SVM performance       |

**Common confusions**: CNG ↔ easybike ↔ leguna, rickshaw ↔ van  
**Rare classes** (like bicycle) consistently underperformed due to severe imbalance.

## Requirements

```bash
pip install -r requirements.txt

Included packages:
torch
torchvision
numpy
matplotlib
scikit-learn
seaborn
pillow
pyyaml

## How to Use

Download dataset
Kaggle: https://www.kaggle.com/datasets/hridoyyahmed/poribohon-bd
Extract to data/poribohon-bd/
Pre-trained models & dataset link (Google Drive)
https://drive.google.com/drive/folders/1mbRH2XQK9ZfPgCg_3jr416xs7mM2ucJ1?usp=drive_link
Preprocess dataset (optional – already handled in loaders)Bashpython preprocess.py
Run full comparison (recommended)
Loads saved models, extracts features, trains/evaluates SVM, shows comparisonsBashpython main_comparison.py
