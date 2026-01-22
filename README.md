🚦 Intelligent Bangladeshi Vehicle Type Classifier 🇧🇩

A Deep Learning Study on Dhaka’s Real-World Traffic

A multi-class image classification system for recognizing 10 native Bangladeshi vehicle types, built on real traffic images from Dhaka.
This project rigorously compares transfer learning vs training from scratch, and deep learning (Softmax) vs classical ML (SVM) on identical visual features.

<p align="center"> <img src="https://github.com/user-attachments/assets/df8a622c-b5fe-4804-812f-dc47a873b370" width="640" alt="Dhaka Traffic Scene"> </p>

🧠 Why This Project?

Most vehicle classifiers are trained on Western datasets (cars, trucks, buses only).
Dhaka traffic is fundamentally different — rickshaws, CNGs, legunas, easybikes dominate the roads.

This project focuses on:

Cultural relevance

Real-world complexity

Practical ML trade-offs

🚘 Vehicle Classes (10)
Category
Bicycle
Bike
Bus
Car
CNG
Easybike
Leguna
Rickshaw
Truck
Van

Key Features & Highlights

📸 Poribohon-BD dataset (~9,058 real-world images)

🧠 Model Architectures

Pretrained ResNet18 (transfer learning)

Custom lightweight CNN (trained from scratch)

⚖️ Classifier Comparison

Softmax (end-to-end deep learning)

SVM on extracted deep features

📊 Handles class imbalance, data augmentation

🔍 Confusion matrix & macro-F1 analysis

🧩 Clean, modular PyTorch + scikit-learn codebase

Project Objectives

Build a Bangladesh-specific vehicle classifier

Quantify transfer learning gains

Compare deep vs classical ML on the same representations

Lay groundwork for Intelligent Transportation Systems (ITS) in Bangladesh

📈 Results Summary
Test Set: 518 Images
Model / ApproachAccuracyMacro-F1Key InsightResNet18 + Softmax80%0.70Best overall performanceCustom CNN + Softmax69%0.59Clear penalty without pretrainingSVM on ResNet18 features55.02%0.50Linear SVM failed to exploit featuresSVM on Custom CNN features~55–60%~0.50Weak features → weak SVM
🔍 Common Confusions


CNG ↔ Easybike ↔ Leguna


Rickshaw ↔ Van


⚠️ Known Limitation


Rare classes (e.g., Bicycle) underperform due to severe imbalance



🧩 Tech Stack


PyTorch – deep learning


Torchvision – pretrained models


scikit-learn – SVM & metrics


Matplotlib / Seaborn – visualization


NumPy, Pillow, PyYAML



📦 Requirements
Install all dependencies:
pip install -r requirements.txt


▶️ How to Use
1️⃣ Download the Dataset
From Kaggle:
🔗 https://www.kaggle.com/datasets/hridoyyahmed/poribohon-bd
Extract to:
data/poribohon-bd/


2️⃣ Download Pre-trained Models
Get saved models and additional files:
🔗 https://drive.google.com/drive/folders/1mbRH2XQK9ZfPgCg_3jr416xs7mM2ucJ1
Place the .pth files in the project root:
resnet_classifier.pth
custom_cnn_classifier.pth


3️⃣ (Optional) Preprocess Dataset
Preprocessing is handled automatically, but you may re-run manually:
python preprocess.py


4️⃣ Run Full Model Comparison (Recommended)
This script:


Loads both trained models


Extracts features


Trains & evaluates SVMs


Prints metrics and confusion matrices


python main_comparison.py


🚀 Future Work


⚖️ Class-weighted loss / SMOTE for imbalance


🔁 Non-linear SVM kernels (RBF, Poly)


🎯 Object detection (YOLO) for multi-vehicle scenes


⏱️ Real-time inference (OpenCV + Flask / Streamlit)


📡 Edge deployment for traffic cameras



🙏 Acknowledgments


Dataset: Poribohon-BD (Tabassum et al., Kaggle / Mendeley Data)


Frameworks: PyTorch, scikit-learn


Inspiration: Dhaka’s vibrant — and chaotic — streets 🚦



⭐ Support
If this project helped your research, coursework, or experimentation,
please consider starring ⭐ the repository.
<p align="center">
  <b>Made with ❤️ in Dhaka 🇧🇩</b>
</p>

Final strategic note (off-README, but important):
If you’re showcasing this for ML internships, research labs, or GitHub portfolio reviews, this README now:


Signals problem ownership


Demonstrates analytical rigor


Reads like a serious applied ML study, not a tutorial clone


