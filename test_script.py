import argparse
import os
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ------------- Argument Parsing -------------
parser = argparse.ArgumentParser(description="Evaluate gender classification model.")
parser.add_argument('--data_path', type=str, required=True, help='Path to test data directory')
parser.add_argument('--model_path', type=str, default='model_balanced.h5', help='Path to trained model')
args = parser.parse_args()

# ------------- Constants -------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------- Load Model -------------
model = load_model(args.model_path)

# ------------- Data Generator -------------
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    args.data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ------------- Predictions -------------
y_true = val_gen.classes
y_probs = model.predict(val_gen).flatten()

# Threshold tuning
best_f1, best_threshold = 0, 0.5
for t in np.arange(0.3, 0.7, 0.01):
    preds = (y_probs >= t).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n✅ Best Threshold: {best_threshold:.2f} | Macro F1: {best_f1:.4f}")
final_preds = (y_probs >= best_threshold).astype(int)

# ------------- Classification Report -------------
class_names = list(val_gen.class_indices.keys())
print("\n📊 Final Classification Report:")
print(classification_report(y_true, final_preds, target_names=class_names, digits=4))

# ------------- Per-Class Accuracy -------------
female_indices = np.where(y_true == 0)[0]
male_indices = np.where(y_true == 1)[0]

female_acc = np.sum(final_preds[female_indices] == y_true[female_indices]) / len(female_indices)
male_acc = np.sum(final_preds[male_indices] == y_true[male_indices]) / len(male_indices)

print(f"\n👩 Female Accuracy: {female_acc*100:.2f}% ({np.sum(final_preds[female_indices] == y_true[female_indices])}/{len(female_indices)})")
print(f"👨 Male Accuracy:   {male_acc*100:.2f}% ({np.sum(final_preds[male_indices] == y_true[male_indices])}/{len(male_indices)})")

# ------------- Confusion Matrix -------------
cm = confusion_matrix(y_true, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ------------- Save Misclassified Images -------------
wrong_indices = np.where(final_preds != y_true)[0]
print(f"\n📂 Saving {len(wrong_indices)} misclassified images...")

for i in wrong_indices:
    img_path = val_gen.filepaths[i]
    true_label = class_names[y_true[i]]
    pred_label = class_names[final_preds[i]]
    
    dest = f"misclassified/{true_label}_as_{pred_label}"
    Path(dest).mkdir(parents=True, exist_ok=True)
    shutil.copy(img_path, os.path.join(dest, os.path.basename(img_path)))

print("✅ Done! Evaluation complete.")
