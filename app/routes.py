from flask import Blueprint, render_template, request, redirect,url_for
import os
from werkzeug.utils import secure_filename
from app.utils import unzip_file
import base64
from app.predict import predict
from PIL import Image
import numpy as np
import io
import base64
import pandas as pd
import cv2
import matplotlib.pyplot as plt


routes = Blueprint('routes', __name__)

# Directories for file uploads
UPLOAD_FOLDER = 'app/uploads'
EXTRACT_FOLDER = 'app/uploads/extracted'
STATIC_FOLDER = 'app/static'
ALLOWED_EXTENSIONS = {'zip'}

import matplotlib
matplotlib.use('Agg')

# Utility function to check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image1(image_array, save_path):
    """Save image using matplotlib."""
    # Create a figure and axis to display the image

    #original image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array[:,:,1])
    plt.axis('off')  # Turn off axis labels and ticks    
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory

def save_image2(image_array,saliency_map,save_path):
    """Save image using matplotlib."""
    # Create a figure and axis to display the image

    #original image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array[:,:,1])
    plt.imshow(saliency_map,alpha=0.75,cmap='viridis')
    plt.axis('off')  # Turn off axis labels and ticks    
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory
    
def save_image3(image_array,grad_cam,save_path):
    """Save image using matplotlib."""
    # Create a figure and axis to display the image

    #original image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array[:,:,1])
    plt.imshow(grad_cam,alpha=0.70,cmap='jet')
    plt.axis('off')  # Turn off axis labels and ticks    
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to free memory    
    
def save_image4(image_array,grad_cam_plus,save_path):
    """Save image using matplotlib."""
    # Create a figure and axis to display the image

    #original image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array[:,:,1])
    plt.imshow(grad_cam_plus,alpha=0.70,cmap='jet')
    plt.axis('off')  # Turn off axis labels and ticks    
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close() 

@routes.route('/')
def home():
    """Home page showing options to upload images."""
    return render_template('index.html')

@routes.route('/model/<model_type>', methods=['GET', 'POST'])
def model_page(model_type):
    """Page for individual model (SAR, S2, RGB)."""
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        # Handle zip file uploads
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Unzip the file and process the images
            try:
                extracted_folder = unzip_file(file_path)
                print(f"Extracted folder: {extracted_folder}")
                print(f"Files in extracted folder: {os.listdir(extracted_folder)}")

                # Preprocess the files and predict
                model_prediction, saliency_map_img, grad_cam_img, grad_cam_plus_img, original_img = predict(extracted_folder, model_type.upper())
                
                original_image_path = os.path.join(STATIC_FOLDER, 'original_image.png')
                saliency_map_path = os.path.join(STATIC_FOLDER, 'saliency_map.png')
                grad_cam_path = os.path.join(STATIC_FOLDER, 'grad_cam.png')
                grad_cam_plus_path = os.path.join(STATIC_FOLDER, 'grad_cam_plus.png')
                
                save_image1(original_img, original_image_path)
                save_image2(original_img,saliency_map_img, saliency_map_path)
                save_image3(original_img,grad_cam_img, grad_cam_path)
                save_image4(original_img,grad_cam_plus_img, grad_cam_plus_path) 
                
                original_image_url = url_for('static', filename='original_image.png')
                saliency_map_url = url_for('static', filename='saliency_map.png')
                grad_cam_url = url_for('static', filename='grad_cam.png')
                grad_cam_plus_url = url_for('static', filename='grad_cam_plus.png')    
                
                          
                # Prepare prediction and images to be displayed
                prediction = 'Flooded' if model_prediction == 1 else 'No Flooding'
                

                return render_template('model_page.html', 
                                    model_type=model_type, 
                                    prediction=prediction,
                                    original_image_url=original_image_url,
                                    saliency_map_url=saliency_map_url, 
                                    grad_cam_url=grad_cam_url,
                                    grad_cam_plus_url=grad_cam_plus_url)
            except FileNotFoundError as e:
                return f"Error: {str(e)}", 400
        
    return render_template('model_page.html', model_type=model_type)


