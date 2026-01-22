# Intelligent Bangladeshi Vehicle Type Classifier

A deep learning-based multi-class image classifier for recognizing **10 native Bangladeshi vehicle types** commonly found in Dhaka's chaotic urban traffic. Built using the **Poribohon-BD** dataset, the project compares **transfer learning (ResNet18)** vs **custom CNN**, and **softmax classification** vs **SVM on extracted features**.

![Dhaka Traffic Example](https://via.placeholder.com/800x300?text=Dhaka+Traffic+Scene)  
*(Replace with a real screenshot or collage of Dhaka traffic / dataset samples)*

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
|----------------------------------------|----------|----------|--------------------------------------------|
| ResNet18 + Softmax (transfer learning) | 80%      | 0.70     | Best overall performance                   |
| Custom CNN + Softmax (from scratch)    | 69%      | 0.59     | Noticeable drop without pre-training       |
| SVM on ResNet18 features               | 55.02%   | 0.50     | Linear kernel underperformed vs softmax    |
| SVM on Custom CNN features             | ~55–60%  | ~0.50    | Weak features → poor SVM performance       |

Common confusions: CNG ↔ easybike ↔ leguna, rickshaw ↔ van  
Rare classes (like bicycle) consistently underperformed due to severe imbalance.


## Requirements

```bash
pip install -r requirements.txt
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

Model & DataSet(Drive)
https://drive.google.com/drive/folders/1mbRH2XQK9ZfPgCg_3jr416xs7mM2ucJ1?usp=drive_link

Preprocess Dataset
python preprocess.py

## Run full comparison
python main_comparison.py

## Future Work Ideas
Class-weighted loss or SMOTE to handle imbalance
RBF/non-linear SVM kernels
Object detection (YOLO) for multi-vehicle scenes
Real-time inference (OpenCV + Flask/Streamlit)
Deployment on edge devices for traffic cameras

## Acknowledgments
Dataset: Poribohon-BD (Tabassum et al., Mendeley Data / Kaggle)
Built with PyTorch, scikit-learn, and inspiration from Dhaka's vibrant (and chaotic) streets

Star ⭐ if this helps your research or studies!
