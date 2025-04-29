from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# Load known faces and their encodings
# Replace these with paths to your known face images
known_image = face_recognition.load_image_file("C:/PythonProject/Image_data/FaceRecognition/Pradeep.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Store known faces and names
known_face_encodings = [known_face_encoding]
known_face_names = ["Person 1"]

print("Known face encoding:", known_face_encoding)
print("Known face name:", known_face_names)


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the frame from the client
        file = request.files['frame']

        frame = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # Resize frame for faster processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #rgb_small_frame = small_frame[:, :, ::-1]  # Convert to RGB
        print("Frame Received1")
        # Perform face detection
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        print("Frame Received2")
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

            # Scale locations back to the original frame size
        scaled_face_locations = [
            (top * 4, right * 4, bottom * 4, left * 4)
            for (top, right, bottom, left) in face_locations
        ]

        # Return detected face names and positions
        return jsonify({"faces": face_names, "locations": scaled_face_locations})

    except Exception as e:
        # Handle exceptions gracefully
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
