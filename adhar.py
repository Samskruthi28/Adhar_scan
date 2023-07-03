from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import socket
import os
import re
import time
import glob
import easyocr
import os
import dotenv
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf

print(tf.__version__)

import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt

import pytesseract
from dotenv import load_dotenv
load_dotenv()
import platform

if platform.system=='windows':
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')


    



app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': '*'}})

# Load Saved Model
cwd = os.getcwd()
PATH_TO_SAVED_MODEL = os.path.join(cwd, "adhar_saved_model")
print('Loading model...', end='')

# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')

#Loading the label_map
PATH_TO_LABELMAP = os.path.join(cwd, 'adhar_data', 'label_map.pbtxt')
category_index=label_map_util.create_category_index_from_labelmap(PATH_TO_LABELMAP,use_display_name=True)

@app.route('/api/scan-adhar', methods=['POST'])
@cross_origin()
def adharscan():
    try:
        if request.method != 'POST':
            return jsonify({ "status": 405, "error": "Method not allowed."}), 405
            
        file = request.files['file']
        if file:
            allowed_files = ['jpg', 'png', 'JPEG', 'jpeg']
            if file.filename.split('.')[-1] not in allowed_files:
                return jsonify({ "status": 400, "error": "File type is not allowed. It must be in jpg, png, or JPEG format."}), 400
            
            im = Image.open(file)

            # Check if image is in portrait mode
            if im.size[0] < im.size[1]:
                # Rotate the image by 90 degrees to switch to landscape mode
                im = im.transpose(Image.ROTATE_90)

            # Save the image
            filename = f'{time.time_ns()}_{file.filename}'
            # file.save('uploads/' + filename)
            im.save('uploads/' + filename)

            def load_image_into_numpy_array(path):
                return np.array(Image.open(path))
                
            image_path = os.path.join(cwd, "uploads", filename)

            image_np = load_image_into_numpy_array(image_path)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)

            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = image_np.copy()
            # print('image with detection-> ', image_np_with_detections)

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.3, # Adjust this value to set the minimum probability boxes to be classified as True
                agnostic_mode=False)

            # APPLY OCR TO DETECTION
            detection_threshold = 0.4
            image = image_np_with_detections
            scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
            boxes = detections['detection_boxes'][:len(scores)]
            classes = detections['detection_classes']
            width = image.shape[1]
            height = image.shape[0]

            results_arr = []

            for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                # if category_index[cls]['name'] == 'cheque':
                #     continue
                roi = box*[height, width, height, width]

                region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]

                if category_index[cls]['name'] == 'adharNo':
                    reader = easyocr.Reader(['en']) 
                    results = reader.readtext(region)[0][1]
                    print(results)


    

                    
                    # result = pytesseract.image_to_string(region, lang='eng')
                    # print(result)
                    # adharNo_pattern = re.compile(r"\b\d{12}\b")
                    # adharNo = adharNo_pattern.search(result)
                    if results:
                        isExists = any(info['label'] == 'adharNo' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'adharNo',
                            'value': results
                        })
                    continue
                
                """ Name """
                if category_index[cls]['name'] == 'name':
                    reader = easyocr.Reader(['en']) 
                    results = reader.readtext(region)[0][1]
                    print(results)
                    # name = pytesseract.image_to_string(region, lang='eng')
                    # print(name)
                    if results:
                        isExists = any(info['label'] == 'name' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'name',
                            'value': results
                            # 'value': name.split('\n').strip()
                        })
                    continue

                """ Account Number """
                


                if category_index[cls]['name'] == 'dob':
                    # reader = easyocr.Reader(['en']) 
                    # results = reader.readtext(region)[0][1]
                    # print(results)
                   
                    txt = pytesseract.image_to_string(region, lang='eng').split(':')[-1].strip()
                    print(txt)
                    # dob_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
                    # accNo = dob_pattern.search(txt)
                    # dob = dob.group()
                    if txt:
                        isExists = any(info['label'] == 'dob' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'dob',
                            'value': txt
                        })
                        # if accNo == '911010049001545':
                        #     results_arr.append({
                        #         'label': 'ifsc',
                        #         'value': 'UTIB0000426'
                        #     })
                    continue

                """ IFSC Number """
                
                if category_index[cls]['name'] == 'gender':
                    reader = easyocr.Reader(['en']) 
                    results = reader.readtext(region)[0][1]
                    print(results)
                    # result = pytesseract.image_to_string(region, lang='eng').split('\n')
                    # print(result)
                    # if 'male' or 'Male' or 'female' or 'FEMALE' in result:
                    #     gender=result[0]

                    if results:
                        isExists = any(info['label'] == 'gender' for info in results_arr)
                        if isExists:
                            continue
                        results_arr.append({
                            'label': 'gender',
                            'value': results
                        })

                """ MICR & Cheque Number """
                # if category_index[cls]['name'] == 'micr':
                #     text = pytesseract.image_to_string(region, lang='mcr')
                #     text = re.sub(r"\s+", "", text)
                    
                #     chequeNo = ''
                #     micr = ''
                    
                #     if 'a' in text:
                #         split_a = text.split('a')[0]
                #         chequeNo = split_a[0:-9]
                #         micr = split_a[-9: len(split_a) - 1]

                #         # # Remove '80' from the beginning of the string, if present
                #         # if chequeNo.startswith("80"):
                #         #     chequeNo = chequeNo[2:]

                #         # # Remove '80' from the end of the string, if present
                #         # if chequeNo.endswith("80"):
                #         #     chequeNo = chequeNo[:-2]
                        
                #         # # Remove 'c' from the beginning of the string, if present
                #         # if chequeNo.startswith("c"):
                #         #     chequeNo = chequeNo[2:]

                #         # # Remove 'c' from the end of the string, if present
                #         # if chequeNo.endswith("c"):
                #         #     chequeNo = chequeNo[:-2]

                #         # # Keep only the first 6 digits
                #         # chequeNo = chequeNo[:6]


                #     isExists = any(info['label'] == 'mcr_text' for info in results_arr)
                #     if isExists:
                #         continue

                #     results_arr.extend([{
                #         'label': 'mcr_text',
                #         'value': text.strip()
                #     }, {
                #         'label': 'chequeNo',
                #         'value': chequeNo.strip()
                #     },{
                #         'label': 'micr',
                #         'value': micr.strip()
                #     }])
                    # continue

            print(results_arr)
            return jsonify({ "status": 200, "data": results_arr}), 200

        else:
            return jsonify({ "status": 500, "error": "Failed to upload file. Please try again."}), 500

    except Exception as e:
        print("ERROR: ", e)
        return jsonify({ "status": 500, "error": "Unable to read the adhar. Please upload clear image again."}), 500
    
if __name__ == '__main__':
    # Get the hostname of the current machine
    hostname = socket.gethostname()
    # Get the IP address of the current machine
    ip_address = socket.gethostbyname(hostname)
    app.run( host=ip_address, port='4000', debug=True)
