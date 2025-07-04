# Task-A-Gender-Classification

This repository implements **Gender Classification** using deep learning with MobileNetV2. The model is trained on augmented and class-balanced data to classify images into **male** or **female** categories with high accuracy.

## Table of Contents
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Visualization](#-visualization)
- [Directory Structure](#-directory-structure)
- [Requirements](#-requirements)

## ✨ Features
- **MobileNetV2** backbone with custom classifier head
- **Focal Loss** implementation for handling class imbalance
- **Data augmentation** with rotation, zoom, shear, and brightness variation
- **Class balancing** through oversampling and custom weights
- **Threshold tuning** for optimal classification
- **Comprehensive evaluation** with precision, recall, F1-score
- **Visualization tools** for model performance

## 🧠 Model Architecture
### Backbone
- Pretrained MobileNetV2 (ImageNet weights)

### Classifier Head
```python
GlobalAveragePooling2D()
Dense(128, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
```

### Loss Function
Focal Loss with parameters:
- α = 0.5
- γ = 1.5

## 🛠 Installation
1. Clone the repository:
```bash
git clone https://github.com/Madhusudan0626/Task-A-Gender-Classification
cd Task-A-Gender-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
### Quick Prediction
```bash
python predict.py --image_path path/to/image.jpg --model_path model_balanced.h5
```

### Full Evaluation
```bash
python test_script.py \
    --data_path /path/to/test_data \
    --model_path model_balanced.h5 \
    --output_dir evaluation_results
```

## 🏋️ Training
To train the model from scratch:
```bash
python train_model.py \
    --data_path /path/to/training_data \
    --epochs 25 \
    --batch_size 32 \
    --output_model model_new.h5
```

### Training Parameters
| Parameter          | Value       |
|--------------------|-------------|
| Input Size         | 224 × 224   |
| Epochs             | 25          |
| Batch Size         | 32          |
| Optimizer          | Adam (1e-5) |
| Early Stopping     | Patience=5  |
| Learning Rate Scheduler | ReduceLROnPlateau |

## 📊 Evaluation
### Expected Test Directory Structure
```
val/
├── male/
└── female/
```

### Output Metrics
The evaluation script generates:
- Classification report (precision, recall, F1-score) 
- Confusion matrix visualization
- Misclassified examples

## 📈 Results
### Final Evaluation Metrics
| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Female  | 0.8947    | 0.8095 | 0.8500   | 105     |
| Male    | 0.9388    | 0.9685 | 0.9534   | 317     |

**Overall Accuracy:** 92.89%  
**Macro F1-Score:** 0.9017

## 📊 Visualization
The repository includes:
1. Training curves (accuracy/loss)
   ![Accuracy Curves](./training_plots/accuracy_curve.png)
   ![Loss Curves](./training_plots/loss_curve.png)
3. Confusion matrix
   
   ![Loss Curves](./evaluation_results/confusion_matrix.png)
   
5. Sample misclassifications
   - Female Incorrectly Predicted as Male
     
    ![!Misclassified_female](./evaluation_results/misclassified/female_as_male/Ai_Sugiyama_0001.jpg)

    ![!Misclassified_female](./evaluation_results/misclassified/female_as_male/Elizabeth_Hill_0001.jpg)
    
   - Male Incorrectly Predicted as Female
     
    ![!Misclassified_male](./evaluation_results/misclassified/male_as_female/Aaron_Peirsol_0001.jpg)

    ![!Misclassified_male](./evaluation_results/misclassified/male_as_female/Michele_Placido_0001.jpg)

7. Model architecture diagram (viewable with [Netron](https://netron.app))
   
   ![Model Architecture](./training_plots/model_architecture.png)
   
## 📁 Directory Structure
```
Gender Classification using MobileNetV2/
├── train_model.py            # Training script
├── test_script.py            # Evaluation script
├── predict.py                # Single image prediction
├── model_balanced.h5         # Pretrained weights
├── requirements.txt          # Dependency list
├── train
├── val
├── evaluation_results/       # Generated during testing
│   ├── confusion_matrix.png
│   └── classification_report.txt
│   └── misclassified/
├── training_plots/           # Generated during training
│   ├── accuracy_curve.png
│   └── loss_curve.png
│   └── model_architecture.png
└── README.md
```

## 📦 Requirements
- Python 3.8+
- TensorFlow 2.9+
- scikit-learn
- matplotlib
- seaborn
- numpy
- OpenCV

Install all dependencies with:
```bash
pip install -r requirements.txt
```
