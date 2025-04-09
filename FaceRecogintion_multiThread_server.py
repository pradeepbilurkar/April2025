from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import threading

app = Flask(__name__)

lock = threading.Lock()
# Load known faces and their encodings
known_image = face_recognition.load_image_file("C:/PythonProject/Image_data/FaceRecognition/Pradeep.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Store known faces and names
known_face_encodings = [known_face_encoding]
known_face_names = ["Pradeep"]

#@app.route('/')
#def home():
#    return "Server is running on the external interface!"

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the frame from the client
        file = request.files['frame']
        frame = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # Perform face detection
        #face_locations = face_recognition.face_locations(frame)
        #face_encodings = face_recognition.face_encodings(frame, face_locations)

        with lock:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)



        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

        # Return detected face names and positions
        return jsonify({"faces": face_names, "locations": face_locations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='192.168.31.51', port=5000, debug=True)
    #app.run(debug=True)
