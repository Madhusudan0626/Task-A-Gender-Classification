
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to test data folder')
args = parser.parse_args()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = load_model("model.h5")

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    args.data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Predict
preds = model.predict(test_generator)
y_true = test_generator.classes
y_pred = (preds > 0.5).astype(int).flatten()

# Metrics
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Female", "Male"]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
