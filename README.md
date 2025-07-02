# 🧑‍🤝‍🧑 Gender Classification using MobileNetV2 – Task A

This repository contains the complete solution for **Task A – Gender Classification** for the **Comys Hackathon**. A modified **MobileNetV2** model is trained and evaluated to classify images into **male** or **female** categories with proper class balancing, focal loss, and data augmentation.

---

## 🧠 Model Architecture

- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Trainable Layers**: Last 30 layers
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.3)`
  - `Dense(1, sigmoid)`
- **Loss**: Focal Loss (α=0.5, γ=1.5)
- **Optimizer**: Adam (LR = 1e-5)
- **Batch Size**: 32
- **Input Size**: (224, 224, 3)

🧩 **Diagram**: The model architecture is saved as `model_architecture.png`.

Place this image at the root of your repo.

Generate using:
```python
from tensorflow.keras.utils import plot_model
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
