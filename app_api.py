import os
import argparse
import requests
import cv2
import numpy as np
import hashlib
import time

from PIL import Image
from flask import Flask, request, make_response, jsonify
from pathlib import Path
from modules import get_prediction
from flask_cors import CORS, cross_origin

parser = argparse.ArgumentParser('Online Food Recognition')
parser.add_argument('--debug', action='store_true', default=False, help="Run app in debug mode")

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = './static/assets/uploads/'

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gpp', '3gp'}

def allowed_file_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS

def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def file_type(path):
    filename = path.split('/')[-1]
    if allowed_file_image(filename):
        filetype = 'image'
    elif allowed_file_video(filename):
        filetype = 'video'
    else:
        filetype = 'invalid'
    return filetype

def save_upload(file):
    """
    Save uploaded image if its format is allowed
    """
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        make_dir(app.config['UPLOAD_FOLDER'])
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return path

    return None

@app.route('/')
def homepage():
    resp = make_response("Hello World")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/analyze', methods=['POST'])
@cross_origin(supports_credentials=True)
def analyze():
    out_name = None
    filepath = None
    filename = None
    filetype = None
    csv_name1 = None
    csv_name2 = None

    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded"})

    file = request.files['file']
    path = save_upload(file)

    if path is None:
        return jsonify({"message": "Invalid file format. Allowed file types: " + ', '.join(IMAGE_ALLOWED_EXTENSIONS)})

    # Get all inputs in form
    iou = request.form.get('threshold-range')
    confidence = request.form.get('confidence-range')
    model_types = request.form.get('model-types')
    enhanced = request.form.get('enhanced')
    ensemble = request.form.get('ensemble')
    tta = request.form.get('tta')
    segmentation = request.form.get('seg')

    ensemble = True if ensemble == 'on' else False
    tta = True if tta == 'on' else False
    enhanced = True if enhanced == 'on' else False
    segmentation = True if segmentation == 'on' else False
    model_types = str.lower(model_types)
    min_conf = float(confidence) / 100
    min_iou = float(iou) / 100

    if allowed_file_image(path):
        # Get filename of detected image
        out_name = "Image Result"
    output_path = os.path.join(
        app.config['UPLOAD_FOLDER'], filename) if not segmentation else os.path.join(
        app.config['SEGMENTATION_FOLDER'], filename)

    output_path, output_type = get_prediction(
        path,
        output_path,
        model_name=model_types,
        tta=tta,
        ensemble=ensemble,
        min_conf=min_conf,
        min_iou=min_iou,
        enhance_labels=enhanced,
        segmentation=segmentation)

filename = os.path.basename(output_path)
csv_name, _ = os.path.splitext(filename)

csv_name1 = os.path.join(
    app.config['CSV_FOLDER'], csv_name + '_info.csv')
csv_name2 = os.path.join(
    app.config['CSV_FOLDER'], csv_name + '_info2.csv')

return jsonify({"filename": filename, "csv_name1": csv_name1, "csv_name2": csv_name2, "output_type": output_type})

if name == 'main':
  if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    args = parser.parse_args()
    app.run(debug=args.debug, use_reloader=False, ssl_context='adhoc')
    
