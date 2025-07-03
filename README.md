# Brain-Tumor-Detection
A deep learning project that uses Convolutional Neural Networks (CNNs) to detect the presence of brain tumors from MRI scans with high accuracy.

---

## 📌 Overview

This project builds a binary classifier using a CNN model to distinguish between:
- MRI scans **with brain tumors**
- MRI scans **without tumors**

It also includes a simple **web interface using Streamlit** where users can upload MRI images and get predictions in real time.

---

## 📂 Project Structure
brain_tumor_detector_project/
├── dataset/ # Organized dataset (yes/no folders)
│ ├── yes/
│ └── no/
├── extracted/ # Auto-generated extracted ZIP (optional)
├── brain_tumor_cnn.py # CNN training script
├── app.py # Streamlit web app
├── organize_dataset.py # Script to unzip and organize folders
├── brain_tumor_detector.h5 # Trained model (auto-generated)
├── .gitignore
└── README.md


---

## 🧪 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Contains 4 classes: `glioma`, `meningioma`, `pituitary`, `notumor`
- This project merges the 3 tumor types into one **binary classification**: `tumor` vs `no tumor`

✅ `organize_dataset.py` automatically organizes the dataset into:

dataset/
├── yes/ # All tumor images
└── no/ # All normal images


---

## 🏗️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain_tumor_detector_project.git
cd brain_tumor_detector_project

pip install -r requirements.txt

python organize_dataset.py (Optional)

python brain_tumor_cnn.py

streamlit run app.py

⚙️ Model Details
- Input size: 150x150 RGB images
- Architecture: 3 convolutional layers + MaxPooling + Dense + Dropout
- Loss: Binary Crossentropy
- Optimizer: Adam
- Accuracy: ~94% on test set

🖼️ Web App (Streamlit)
Users can upload an MRI image and get real-time classification:
