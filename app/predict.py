import tensorflow as tf
from app.utils import preprocess_image,get_saliency,get_gradcam,get_gradcam_plus
from tf_keras_vis.utils.scores import CategoricalScore
import numpy as np



def load_model_for_prediction(model_type):
    """Load the appropriate model for prediction."""
    if model_type == 'SAR':
        model = tf.keras.models.load_model('CNN_models/SAR_CNN.h5')  # Example SAR model path
    elif model_type == 'S2':
        model = tf.keras.models.load_model('CNN_models/S2_CNN.h5')  # Example S2 model path
    elif model_type == 'RGB':
        model = tf.keras.models.load_model('CNN_models/RGB_CNN.h5')  # Example RGB model path
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    return model

def predict(file_path, model_type):
    """Make prediction on the uploaded image and generate heatmap."""
    # Load and preprocess the image
    image = preprocess_image(file_path, model_type)
    image = image.astype(np.float32)
    
    # Load the model
    model = load_model_for_prediction(model_type)
    
    # Make prediction
    pred = model.predict(image[None, ...])  # Add batch dimension
    model_prediction = np.argmax(pred, axis=-1)[0]  # 0: No Flooding, 1: Flooded
    score = CategoricalScore(model_prediction)
    # Generate the heatmap
    saliency_map_img = get_saliency(image[None, ...], score, model)
    grad_cam_img = get_gradcam(image[None, ...], score, model)
    grad_cam_plus_img = get_gradcam_plus(image[None, ...], score, model)

    return model_prediction, saliency_map_img, grad_cam_img, grad_cam_plus_img, image
