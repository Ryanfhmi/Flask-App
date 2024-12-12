from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

bp = Blueprint('main', __name__)

# Load your trained model
model = tf.keras.models.load_model('final_model_quantized.tflite')
labels = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

@bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        # Process the uploaded image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((150, 150))
        img_array = np.array(img)[:, :, :3] / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_label = labels[np.argmax(prediction)]

        return jsonify({'result': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500