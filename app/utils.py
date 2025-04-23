import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import cm
import rasterio as rio
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from PIL import Image

def unzip_file(file_path):
    """Unzip a file and return the folder path where files are extracted."""
    extracted_folder = file_path.rsplit('.', 1)[0]  # Path to extracted folder
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    # Check if there is a subfolder inside the extracted folder
    extracted_files = os.listdir(extracted_folder)
    if len(extracted_files) == 1 and os.path.isdir(os.path.join(extracted_folder, extracted_files[0])):
        # If there's only one subfolder, return its path
        extracted_folder = os.path.join(extracted_folder, extracted_files[0])

    return extracted_folder

def read_tif_image(file_path):
    try:
        img = Image.open(file_path)
        img = np.array(img)  # Convert to numpy array
        return img
    except Exception as e:
        print(f"Failed to load image {file_path}: {e}")
        return None

def preprocess_image(file_path, model_type):
    """Preprocess images for SAR, S2, or RGB model types."""


    def read_raster(file_path):
        """Read .tif file using rasterio."""
        with rio.open(file_path) as f:
            return f.read(1)  # Read the first band

    # List all .tif files in the given directory
    images = [os.path.join(file_path, fname) for fname in os.listdir(file_path) if fname.endswith('.tif')]
    if not images:
        raise ValueError(f"No .tif files found in {file_path} for model type {model_type}.")
    
    if model_type == 'SAR':
        # Expect two bands: VH and VV
        if len(images) < 2:
            raise ValueError(f"Not enough images for SAR data. Expected 2, found {len(images)}.")
        images = sorted(images)  # Ensure correct order
        
        vh = read_raster(images[0]) / 50.0  # Normalize using SAR-specific scaling
        vv = read_raster(images[1]) / 100.0
        
        image = np.stack([vh, vv], axis=-1)
        image = cv2.resize(image, (256, 256))
        return image

    elif model_type == 'S2':
        bands = []
        for band_file in sorted(images):  # Ensure order
            band = read_raster(band_file) / 10000.0  # Normalize
            band = cv2.resize(band, (256, 256))
            bands.append(band)
        if len(bands) != 12:  # Ensure correct raster count
            raise ValueError(f"Expected 12 bands for S2 data, found {len(bands)}.")
        return np.stack(bands, axis=-1)

    elif model_type == 'RGB':
        required_bands = ['B02', 'B03', 'B04']
        band_files = {band: next((f for f in images if band in os.path.basename(f)), None) for band in required_bands}
        missing_bands = [band for band, f in band_files.items() if f is None]
        if missing_bands:
            raise ValueError(f"Missing required bands for RGB: {', '.join(missing_bands)}")

        rgb_bands = []
        for band_file in band_files.values():
            band = read_raster(band_file) / 10000.0  # Normalize for RGB bands
            band = cv2.resize(band, (256, 256))
            rgb_bands.append(band)
        return np.stack(rgb_bands, axis=-1)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model Modifier Function
def model_modifier_function(cloned_model):
    if hasattr(cloned_model.layers[-1], "activation"):
        cloned_model.layers[-1].activation = tf.keras.activations.linear


# Saliency Map
def get_saliency(img, score, cnn_model):
    img = img.astype(np.float32)
    saliency = Saliency(cnn_model, model_modifier=model_modifier_function, clone=True)
    sal_map = saliency(score, img, smooth_samples=20, smooth_noise=0.20)
    return sal_map.squeeze(axis=0)  # Remove batch dimension


# Grad-CAM
def get_gradcam(img, score, cnn_model):
    gradcam = Gradcam(cnn_model, model_modifier=model_modifier_function, clone=True)
    cam = gradcam(score, img, seek_penultimate_conv_layer=True)
    cam = np.maximum(cam, 0)  # ReLU-like activation
    cam = cam / cam.max() if cam.max() > 0 else cam  # Normalize to [0, 1]
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Convert to RGB heatmap
    return heatmap


# Grad-CAM++
def get_gradcam_plus(img, score, cnn_model):
    gradcam_plus = GradcamPlusPlus(cnn_model, model_modifier=ReplaceToLinear(), clone=True)
    cam = gradcam_plus(score, img)
    cam = np.maximum(cam, 0)  # ReLU-like activation
    cam = cam / cam.max() if cam.max() > 0 else cam  # Normalize to [0, 1]
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Convert to RGB heatmap
    return heatmap








"""
def predict_and_visualize(extracted_folder, model, model_type, class_dict):
    
    preprocessed_image = preprocess_image(extracted_folder, model_type)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(prediction)

    # Create Score
    score = CategoricalScore(predicted_class)

    # Generate Visualizations
    saliency_map_img = get_saliency(np.expand_dims(preprocessed_image, axis=0), score, model)
    grad_cam_img = get_gradcam(np.expand_dims(preprocessed_image, axis=0), score, model)
    grad_cam_plus_img = get_gradcam_plus(np.expand_dims(preprocessed_image, axis=0), score, model)

    return predicted_class, saliency_map_img, grad_cam_img, grad_cam_plus_img """
