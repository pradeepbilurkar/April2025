from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def process_frame(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 5), 0)
        edges = cv2.Canny(blur, 40, 165)

        # Define ROI
        height, width = image.shape[:2]
        roi_vertices = [
            (0, int(height * 0.7)),
            (width - 1, int(height * 0.7)),
            (width - 1, height - 1),
            (0, height - 1) 
        ]
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        # Hough Transform
        lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi/60, threshold=90, minLineLength=25, maxLineGap=45)
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                print(x1,y1,x2,y2)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Combine lines with the original image
        final_image = cv2.addWeighted(image, 0.9, line_image, 2, 1)
        _, buffer = cv2.imencode('.jpg', final_image)
        return buffer.tobytes()
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        file = request.files['video']
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False)  # Create temporary file
        file.save(temp_file.name)  # Save the uploaded video to the temporary file
        cap = cv2.VideoCapture(temp_file.name)  # Load the video file

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            if processed_frame:
                frames.append(processed_frame)
        cap.release()
        return jsonify({"frames": [frame.decode('latin1') for frame in frames]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

