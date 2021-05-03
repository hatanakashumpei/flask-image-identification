# coding: utf-8

"""
this code is to upload files
@ shumpei hatanaka
"""

import sys
import os
from flask import Flask, request, flash, redirect, url_for
from flask import send_from_directory
from numpy.lib.function_base import percentile
from werkzeug.utils import secure_filename
# keras
from keras.models import Sequential, load_model
import numpy as np
from PIL import Image
from generate_data import get_classes
# used the following link as a reference
# https://github.com/tensorflow/tensorflow/issues/24496
from keras import backend as K
if 'tensorflow' == K.backend():
    import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
# tf.compat.v1.keras.backend.set_session(tf.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image_label(filepath):
    model = load_model("./model/model_aug.h5")
    image = Image.open(filepath)
    image = image.convert("RGB")
    image = image.resize((50, 50))
    data = np.asarray(image) / 255
    x = []
    x.append(data)
    x = np.array(x)

    result = model.predict([x])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)

    return predicted, percentage

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # remove dangerous characters
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # save file
            file.save(filepath)
            # predict image
            classes = get_classes()
            predicted, percentage = predict_image_label(filepath)
            # go to the page after uploading.
            # return redirect(url_for('uploaded_file', filename=filename))

            return f"{classes[predicted]} {percentage} %"

    return '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Upload new File</title>
        </head>
        <body>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
            </form>
        </body>
    </html>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
