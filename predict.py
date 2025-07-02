import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def focal_loss(alpha=0.5, gamma=1.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return tf.reduce_mean(
            -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
            - (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        )
    return loss

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_gender(model, img_array, threshold=0.55):
    prob = model.predict(img_array)[0][0]
    gender = "female" if prob < threshold else "male"
    confidence = (1 - prob) * 100 if gender == "female" else prob * 100
    return gender, confidence, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gender Classification Prediction")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='model_balanced.h5', help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.55, help='Classification threshold')
    args = parser.parse_args()

    model = load_model(args.model_path, custom_objects={'loss': focal_loss(alpha=0.5, gamma=1.5)})
    img_array = load_and_preprocess(args.image_path)
    
    gender, confidence, prob = predict_gender(model, img_array, args.threshold)
    
    print(f"\nPrediction: {gender} (Confidence: {confidence:.2f}%)")
    print(f"Raw Probability: {prob:.4f}")
    
    # Display the image
    img = image.load_img(args.image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {gender} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()
