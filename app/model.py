import os
import zipfile
import numpy as np
import tensorflow as tf
from app.utils import preprocess_image

def extract_zip(zip_path, extract_to):
    """Extract the zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_model_for_prediction(model_type):
    """Load the appropriate model for prediction."""
    if model_type == 'SAR':
        model = tf.keras.models.load_model('models/sar_cnn.h5')  # Example SAR model path
    elif model_type == 'S2':
        model = tf.keras.models.load_model('models/s2_cnn.h5')  # Example S2 model path
    elif model_type == 'RGB':
        model = tf.keras.models.load_model('models/rgb_cnn.h5')  # Example RGB model path
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    return model

def predict(zip_file_path, model_type):
    """Make prediction on the images from the uploaded zip file."""
    # Create a temporary folder to extract images
    extract_folder = 'app/uploads/extracted_images'
    os.makedirs(extract_folder, exist_ok=True)
    
    # Extract zip file
    extract_zip(zip_file_path, extract_folder)
    
    # Get all .tif files from the extracted folder
    tif_files = [os.path.join(extract_folder, f) for f in os.listdir(extract_folder) if f.lower().endswith('.tif')]
    
    if len(tif_files) == 0:
        raise ValueError("No .tif files found in the uploaded zip file.")
    
    # Load the model
    model = load_model_for_prediction(model_type)
    
    # Preprocess all images
    preprocessed_images = [preprocess_image([tif], model_type) for tif in tif_files]
    
    # Stack all preprocessed images (if multiple)
    preprocessed_images = np.stack(preprocessed_images, axis=0)
    
    # Make prediction
    pred = model.predict(preprocessed_images)
    return pred
