import cv2
import requests
import numpy as np

# API Endpoint
url = "http://127.0.0.1:5000/process_frame"


def send_frame(frame):
    # Encode the frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    response = requests.post(url, files={'frame': buffer.tobytes()})

    if response.status_code == 200:
        # Decode the received processed frame
        processed_frame = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)
    else:
        print(f"Error from server: {response.json()}")
        return None


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Send frame to server and get the processed frame
        processed_frame = send_frame(frame)
        if processed_frame is not None:
            cv2.imshow("Lane Detection", cv2.resize(processed_frame, (300, 360)))

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video("C:/PythonProject/VID_20230906_174106981_Trim.mp4")
