import os
import argparse
import requests
import cv2
import numpy as np
import hashlib
import time

from PIL import Image
from flask import Flask, request, render_template, redirect, make_response, jsonify
from pathlib import Path
from werkzeug.utils import secure_filename
from modules import get_prediction
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__, template_folder='templates', static_folder='static')
app.config["NGROK"] = True
app.config["HOST"] = "localhost:8000"
app.config["DEBUG"] = False

CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

UPLOAD_FOLDER = './static/assets/uploads/'
CSV_FOLDER = './static/csv/'
SEGMENTATION_FOLDER = './static/assets/segmentations/'
DETECTION_FOLDER = './static/assets/detections/'
METADATA_FOLDER = './static/metadata/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER
app.config['SEGMENTATION_FOLDER'] = SEGMENTATION_FOLDER

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gpp', '3gp'}


def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS


def allowed_file_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS


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


def download(url):
    """
    Handle input url from client 
    """

    make_dir(app.config['UPLOAD_FOLDER'])
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2)',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
    r = requests.get(url, stream=True, headers=headers)
    print('Image Url')

    # Get cache name by hashing image
    data = r.content
    ori_filename = url.split('/')[-1]
    _, ext = os.path.splitext(ori_filename)
    filename = hashlib.sha256(data).hexdigest() + f'{ext}'

    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with open(path, "wb") as file:
        file.write(r.content)

    return filename, path


def save_upload(file):
    """
    Save uploaded image and video if its format is allowed
    """
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        make_dir(app.config['UPLOAD_FOLDER'])
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    elif allowed_file_video(filename):
        make_dir(app.config['VIDEO_FOLDER'])
        path = os.path.join(app.config['VIDEO_FOLDER'], filename)

    file.save(path)

    return path


path = Path(__file__).parent


@app.route('/')
def homepage():
    resp = make_response(render_template("upload-file.html"))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/url')
def detect_by_url_page():
    return render_template("input-url.html")


@app.route('/webcam')
def detect_by_webcam_page():
    return render_template("webcam-capture.html")


@app.route('/analyze', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def analyze():
    if request.method == 'POST':
        out_name = None
        filepath = None
        filename = None
        filetype = None
        csv_name1 = None
        csv_name2 = None

        print("File: ", request.files)

        if 'webcam-button' in request.form:
            # Get webcam capture

            f = request.files['blob-file']
            ori_file_name = secure_filename(f.filename)
            filetype = file_type(ori_file_name)

            filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # save file to /static/uploads
            img = Image.open(f.stream)
            img.save(filepath)

        elif 'url-button' in request.form:
            # Get image/video from input url

            url = request.form['url_link']
            filename, filepath = download(url)

            filetype = file_type(filename)

        elif 'upload-button' in request.form:
            # Get uploaded file

            f = request.files['file']
            ori_file_name = secure_filename(f.filename)
            _, ext = os.path.splitext(ori_file_name)

            filetype = file_type(ori_file_name)

            if filetype == 'image':
                # Get cache name by hashing image
                data = f.read()
                filename = hashlib.sha256(data).hexdigest() + f'{ext}'

                # Save file to /static/uploads
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                np_img = np.fromstring(data, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                cv2.imwrite(filepath, img)

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
        min_conf = float(confidence)/100
        min_iou = float(iou)/100

        if filetype == 'image':
            # Get filename of detected image
            out_name = "Image Result"
            output_path = os.path.join(
                app.config['DETECTION_FOLDER'], filename) if not segmentation else os.path.join(
                app.config['SEGMENTATION_FOLDER'], filename)

            output_path, output_type = get_prediction(
                filepath,
                output_path,
                model_name=model_types,
                tta=tta,
                ensemble=ensemble,
                min_conf=min_conf,
                min_iou=min_iou,
                enhance_labels=enhanced,
                segmentation=segmentation)

        else:
            error_msg = "Invalid input url!!!"
            return render_template('detect-input-url.html', error_msg=error_msg)
