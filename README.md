Markdown# Intelligent Bangladeshi Vehicle Type Classifier

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

Install dependencies:

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

How to Use
1. Download Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/hridoyyahmed/poribohon-bd
Extract the zip file to:
data/poribohon-bd/
2. Download Pre-trained Models & Additional Dataset Files
Get the saved models (resnet_classifier.pth, custom_cnn_classifier.pth) and any extra dataset files:
https://drive.google.com/drive/folders/1mbRH2XQK9ZfPgCg_3jr416xs7mM2ucJ1?usp=drive_link
Place the .pth files in the project root folder (same level as main_comparison.py).
3. Preprocess Dataset (Optional)
Preprocessing is already handled inside the data loaders, but if you want to verify or re-run manually:
Bashpython preprocess.py
4. Run Full Comparison (Recommended)
This script does everything:

Loads saved ResNet18 and Custom CNN models
Extracts features from both
Trains and evaluates SVM on both feature sets
Prints detailed comparison tables, accuracies, and confusion matrices

Bashpython main_comparison.py
Future Work Ideas

Class-weighted loss or SMOTE to handle imbalance
RBF/non-linear SVM kernels
Object detection (YOLO) for multi-vehicle scenes
Real-time inference (OpenCV + Flask/Streamlit)
Deployment on edge devices for traffic cameras

Acknowledgments

Dataset: Poribohon-BD (Tabassum et al., Mendeley Data / Kaggle)
Built with PyTorch, scikit-learn, and inspiration from Dhaka's vibrant (and chaotic) streets

Star ⭐ if this helps your research or studies!
Made with ❤️ in Dhaka
textThis version:
- Keeps **Requirements** only for dependencies
- Moves **How to Use** to its own major section
- Gives each step its own numbered sub-heading (1., 2., 3., 4.)
- Uses clear, numbered instructions for easy following
- Looks clean and professional on GitHub
