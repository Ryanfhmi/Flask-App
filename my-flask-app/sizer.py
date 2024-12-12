import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model("final_model.h5")

# Convert to TFLite with optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the compressed model
with open("final_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)
