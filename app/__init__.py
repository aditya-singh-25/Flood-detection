from flask import Flask
import os
from app.routes import routes  # Import the routes blueprint

def create_app():
    # Create Flask app
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    # Configurations
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')  # Folder for ZIP uploads
    app.config['EXTRACT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads/extracted')  # Folder to extract ZIP files
    
    # Ensure the folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)
    
    # Register the routes blueprint
    app.register_blueprint(routes)  # This should match the import in your routes.py file

    return app
