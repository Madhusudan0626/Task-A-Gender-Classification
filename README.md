# 🧑‍🤝‍🧑 Gender Classification using MobileNetV2

This repository contains the implementation of **Task A – Gender Classification** for the Comys Hackathon. The model uses **MobileNetV2** (pretrained on ImageNet) and classifies images into **male** or **female**.

---

## ✅ Submission Guidelines

- ✔️ Training and Validation Results with accuracy, precision, recall, and F1-score
- ✔️ Well-documented source code
- ✔️ Pretrained model weights (`model.h5`)
- ✔️ Test script (`test_script.py`) to evaluate new test data
- ✔️ This `README.md` file

---

## 📁 Project Directory Structure

```text
├── train/                    # Training images ('male/', 'female/')
├── val/                      # Validation images ('male/', 'female/')
├── model.h5                  # Pretrained model weights
├── gender_classification.py  # Main training code
├── test_script.py            # Evaluation script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🧠 Model Architecture

- Base Model: MobileNetV2 (frozen, pretrained on ImageNet)
- Input Shape: (224, 224, 3)
- Layers:
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(64, activation='relu')
  - Dense(1, activation='sigmoid')
- Loss: Binary Crossentropy
- Optimizer: Adam (learning rate = 0.0001)
- Epochs: 10
- Batch Size: 32

---

## 📊 Model Performance

**Confusion Matrix:**

```text
          Predicted
          Female  Male
Actual
Female      63     42
Male         7    310
```

**Classification Report:**

| Class   | Precision | Recall | F1-score | Support |
|---------|-----------|--------|----------|---------|
| Female  |   0.90    |  0.60  |   0.72   |   105   |
| Male    |   0.88    |  0.98  |   0.93   |   317   |
| Accuracy|     —     |   —    | **0.88** |   422   |
| Macro avg | 0.89    | 0.79   | 0.82     |   422   |
| Weighted avg | 0.89 | 0.88   | 0.88     |   422   |

---

## 🔧 How to Train the Model

```bash
python gender_classification.py
```

This will:
- Load and augment data from `train/` and `val/`
- Train the MobileNetV2 model
- Save the model as `model.h5`

---

## 🧪 How to Evaluate on Test Data

Ensure test data is structured like:

```text
test_data/
├── male/
├── female/
```

Run the evaluation script:

```bash
python test_script.py --data_path /path/to/test_data
```

The script will:
- Load `model.h5`
- Evaluate on test data
- Print accuracy, precision, recall, F1-score, and confusion matrix

---

## 💾 Load Pretrained Model

```python
from tensorflow.keras.models import load_model
model = load_model("model.h5")
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```text
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🧩 Notes

- MobileNetV2 chosen for its efficiency and speed
- Data augmentation improves generalization
- Better recall for male class; female classification may improve with:
  - Class balancing
  - Fine-tuning deeper layers
  - More data

---

## 📧 Contact

**Author**: Madhusudan Chand  
📧 Email: [your_email@example.com]  
🔗 GitHub: [https://github.com/yourusername/gender-classification]
