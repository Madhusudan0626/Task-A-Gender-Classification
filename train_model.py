import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

# ------------- Paths and Configs -------------
train_dir = "/home/madhusudanxsoul/Downloads/Comys_Hackathon5/Task_A/train"
val_dir = "/home/madhusudanxsoul/Downloads/Comys_Hackathon5/Task_A/val"
img_size = (224, 224)
batch_size = 32
seed = 42

# ------------- Oversample Female -------------
# Simple oversampling: duplicate female directory into a temp folder
oversampled_train_dir = "/tmp/Task_A_balanced_train"
os.makedirs(oversampled_train_dir, exist_ok=True)

# Copy male data as is
if not os.path.exists(os.path.join(oversampled_train_dir, 'male')):
    shutil.copytree(os.path.join(train_dir, 'male'), os.path.join(oversampled_train_dir, 'male'))

# Copy female data twice (oversample)
female_target = os.path.join(oversampled_train_dir, 'female')
if not os.path.exists(female_target):
    os.makedirs(female_target, exist_ok=True)
    female_src = os.path.join(train_dir, 'female')
    for i in range(2):  # duplicate female samples
        for fname in os.listdir(female_src):
            src_path = os.path.join(female_src, fname)
            dst_path = os.path.join(female_target, f"{i}_{fname}")
            shutil.copy(src_path, dst_path)

# ------------- Data Generators -------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_directory(
    oversampled_train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    seed=seed
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ------------- Focal Loss -------------
def focal_loss(alpha=0.5, gamma=1.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(
            -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
            - (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        )
    return loss

# ------------- Model Definition -------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(alpha=0.5, gamma=1.5),
    metrics=['accuracy']
)

model.summary()

# ------------- Class Weights -------------
labels = train_gen.classes
cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw))
class_weights[0] *= 1.0  # female
class_weights[1] *= 0.9  # male
print("Class weights:", class_weights)

# ------------- Callbacks -------------
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

# ------------- Training -------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

model.save("model_balanced.h5")

# ------------- Training Curves -------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend(); plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend(); plt.show()
