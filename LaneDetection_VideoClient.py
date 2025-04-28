import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# API Endpoint
url = "http://localhost:5000/process_video"

def send_video(file_path):
    with open(file_path, 'rb') as video_file:
        response = requests.post(url, files={'video': video_file})
        if response.status_code == 200:
            data = response.json()
            frames = data.get("frames", [])
            for frame_data in frames:
                frame = np.frombuffer(frame_data.encode('latin1'), dtype=np.uint8)
                image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (300, 400))
                cv2.imshow("Processed Frame", image)
                if cv2.waitKey(10) & 0xFF == 27:  # ESC key to exit
                    break
            cv2.destroyAllWindows()
        else:
            print(f"Error: {response.json()}")

if __name__ == "__main__":
    send_video("C:/PythonProject/VID_20230906_174106981_Trim.mp4")
