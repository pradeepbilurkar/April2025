from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)


def detect_lanes(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny Edge Detector
        edges = cv2.Canny(blur, 50, 150)

        # Define Region of Interest (ROI)
        height, width = image.shape[:2]
        roi_vertices = [
            (width // 8, height),
            (width * 7 // 8, height),
            (width // 2, height // 3)
        ]

        top_boundary = int(height * 0.8)  # Start at 80% of the height
        roi_vertices = [
            (0, top_boundary),  # Top-left corner
            (width - 1, top_boundary),  # Top-right corner
            (width - 1, height - 1),  # Bottom-right corner
            (0, height - 1)  # Bottom-left corner
        ]

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)

        # Draw lanes on a blank image
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Overlay lines on the original frame
        combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0)
        return combined_image
    except Exception as e:
        print(f"Error detecting lanes: {e}")
        return None


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Receive the frame from the client
        file = request.files['frame']
        np_frame = np.frombuffer(file.read(), dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

        # Process the frame for lane detection
        processed_frame = detect_lanes(frame)
        if processed_frame is None:
            return jsonify({"error": "Failed to process frame"}), 500

        # Encode the processed frame into a byte stream
        resized_frame = cv2.resize(processed_frame, (300, 360))

        _, buffer = cv2.imencode('.jpg', processed_frame)
        return buffer.tobytes(), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
