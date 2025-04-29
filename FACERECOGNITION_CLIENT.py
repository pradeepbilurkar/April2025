import cv2
import requests

API_URL = "http://127.0.0.1:5000/api/process_frame"

def main():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Encode the frame as a JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        # Send the frame to the server
        response = requests.post(API_URL, files={"frame": frame_data})
        print(response.status_code)
        if response.status_code == 200:
            data = response.json()
            face_names = data["faces"]
            face_locations = data["locations"]
            print(face_names)
            # Draw rectangles and labels around detected faces
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                #cv2.putText(frame, face_names[0], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, face_names[0], (200, 300 ), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),
                            1)
            cv2.imshow("Live Video Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
