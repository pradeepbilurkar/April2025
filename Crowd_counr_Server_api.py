from flask import Flask, request, send_file
import os
import cv2
import numpy as np
#from your_module import run_crowd_count  # Your existing crowd-counting function
from lwcc import LWCC

app = Flask(__name__)

# Set up directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the image
    processed_file_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    count = run_crowd_count(file_path, processed_file_path)

    # Send the processed file back as a response
    return send_file(processed_file_path, mimetype='image/jpeg')

# Define crowd-counting function for processing
def run_crowd_count(input_path, output_path):
    img = cv2.imread(input_path)
    count = LWCC.get_count(input_path, model_name="DM-Count", model_weights="SHB")
    count = int(np.round(count, 0))
    cv2.putText(img, f"Number of Persons: {count}", (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.imwrite(output_path, img)
    return count

if __name__ == '__main__':
    app.run(debug=True)
