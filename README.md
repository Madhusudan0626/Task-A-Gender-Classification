# 🧑‍🤝‍🧑 Gender Classification using MobileNetV2

This repository implements **Task A – Gender Classification** using deep learning. It leverages **MobileNetV2**, trained on augmented and class-balanced data, to classify images into **male** or **female** categories with high accuracy.

---

## 📁 Project Structure
```
├── train_model.py # Training script with augmentation and oversampling
├── test_script.py # Evaluation script with threshold tuning
├── model_balanced.h5 # Pretrained model weights
├── model_architecture.png # Visual diagram of the model
├── confusion_matrix.png # Confusion matrix plot
├── Screenshot-*.png # Training plots (accuracy/loss)
├── misclassified/ # Saved misclassified validation images
└── README.md
```
---

## 🧠 Model Architecture

- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Classifier Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(1, activation='sigmoid')`
- **Loss Function**: Focal Loss (α = 0.5, γ = 1.5)

### 🔍 Visualizing Model Diagram with Netron

You can view the model architecture using [**Netron**](https://netron.app):

1. Go to: https://netron.app  
2. Drag and drop `model_balanced.h5`

### ⚙️ Training Configuration

    Input Size: 224 × 224

    Epochs: 25

    Batch Size: 32

    Optimizer: Adam (learning rate = 1e-5)

    EarlyStopping + ReduceLROnPlateau callbacks

### 📈 Data Augmentation

    Rotation, zoom, shear

    Width/height shift

    Brightness variation

    Horizontal flip

### ⚖️ Class Balancing

    Female class was oversampled

    Custom class weights applied:
    { female: 1.0, male: 0.9 }

### 📊 Final Evaluation Metrics

- ✅ Best Threshold: 0.55
- 🎯 Macro F1-Score: 0.9017
Class	Precision	Recall	F1-Score	Support
Female	0.8947	0.8095	0.8500	105
Male	0.9388	0.9685	0.9534	317
Accuracy	0.9289			422
Macro Avg	0.9168	0.8890	0.9017	
Weighted Avg	0.9279	0.9289	0.9277	

    👩 Female Accuracy: 80.95% (85/105)

    👨 Male Accuracy: 96.85% (307/317)

### 📉 Training Curves
Accuracy	Loss
	
🔎 Confusion Matrix

Confusion Matrix

### 🚀 Inference & Testing

Evaluate the model on new data using the test_script.py:

python test_script.py --data_path /path/to/test_data --model_path model_balanced.h5

### 📥 Expected Test Directory Structure

test_data/
├── male/
└── female/

### 📤 Outputs

    ✅ Threshold-optimized classification report

    📊 Confusion matrix (confusion_matrix.png)

    ❌ Misclassified images saved to misclassified/ directory

### 💾 Pretrained Model

The trained model is saved as:

model_balanced.h5

It will automatically load in test_script.py.

### 📦 Requirements

Ensure the following dependencies are installed:

pip install tensorflow>=2.9 scikit-learn seaborn matplotlib numpy
