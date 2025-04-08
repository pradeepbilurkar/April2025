import warnings
warnings.filterwarnings("ignore")
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QFrame,QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QPixmap, QImage,QColor
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon , QFont
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtGui import QKeyEvent
from sympy import false
import image_dehazer
import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
import face_recognition as fr
import pickle
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import argparse
import imutils
import time
import dlib
import os
from pygame import mixer
import keyboard
import numpy as np
from deepface import DeepFace
import os
os.add_dll_directory("C:/Users/Samsan/anaconda3/DLLs")
from lwcc import LWCC
#from Silent_Face_Anti_Spoofing_master import test1
import argparse
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import speech_recognition as sr
import threading


mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')
Image_Height = 350

class CircularProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 50  # Default value, to be updated with system volume

    def setProgress(self, value):
        self.progress = value
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Define size and center
        rect = self.rect()
        center_x, center_y = rect.width() // 2, rect.height() // 2
        radius = min(rect.width(), rect.height()) // 2 - 10

        # Draw background circle
        pen_bg = QPen(Qt.lightGray, 10)
        painter.setPen(pen_bg)
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # Draw progress arc
        pen_fg = QPen(Qt.blue, 10)
        painter.setPen(pen_fg)
        start_angle = -90 * 16  # Start at the top
        span_angle = int(360 * (self.progress / 100)) * 16  # Fill based on progress
        painter.drawArc(center_x - radius, center_y - radius, radius * 2, radius * 2, start_angle, span_angle)
        # Draw percentage in the center
        painter.setPen(Qt.black)
        painter.setFont(painter.font())
        painter.drawText(rect, Qt.AlignCenter, f"{self.progress}%")


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SAMSAN V-SENSE")
        self.setStyleSheet("background-color: white;")
        self.setFixedSize(1360, 650)

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Top layout for logo
        self.top_layout = QHBoxLayout()
        self.logo = QLabel()  # Logo label
        #self.logo_pixmap = QPixmap("C:/Users/Samsan/Downloads/SAM_SANVAD2.GIF")
        self.logo_pixmap = QPixmap("C:/Users/Samsan/Desktop/logo.png")
        # Replace with your company logo file path
        self.logo.setPixmap(self.logo_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignLeft)  # Align logo to the left
        self.top_layout.addWidget(self.logo)
        self.main_layout.addLayout(self.top_layout)

        # Bottom layout
        self.bottom_layout = QHBoxLayout()

        # First panel: Buttons
        self.first_panel = QFrame()
        self.first_panel.setStyleSheet("background-color: #bbbbbb; border-radius: 10px;")
        self.first_layout = QVBoxLayout(self.first_panel)

        # Add spacing in the first panel
        self.first_layout.addStretch()

        # Initialize paths and labels for uploaded images
        self.image1 = None  # For image1
        self.image2 = None
        self.image3= None# For image2
        self.image_labels = {}

        self.cap = None  # Video capture for the right panel
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.speech_thread = None  # Speech recognition
        self.volume_control = self.getVolumeControl()
        self.current_volume = int(self.volume_control.GetMasterVolumeLevelScalar() * 100)  #
        self.speech_running = False

        self.cap_recorded = None
        self.timer_recorded = QTimer()
        self.timer_recorded.timeout.connect(self.update_laneDeparture_video)

        # Add functional buttons to the first panel
        button_names = ["Stitch Image", "Remove Fog", "Remove Background", "Change Background", "Meeting Background Change", "Face Recognition", "Face Antisoofing", "DMS"
                        , "Lane Departure","Crowd Count", "Speech Recognition", "Exit"]
        for name in button_names:
            button = QPushButton(name)
            #button.setStyleSheet("font-size: 40px ") # ;font-weight: bold;
            #button.setStyleSheet("QPushButton:hover {background-color: red;}")
            #button.setFixedSize(100, 80)
            if name == "Stitch Image":  # Edge detection functionality
                button.clicked.connect(self.stitch_images)
            elif name == "Remove Fog":  # Grayscale functionality
                button.clicked.connect(self.Fog_Remove)
            elif name == "Remove Background":  # Grayscale functionality
                button.clicked.connect(self.Remove_Background)
            elif name == "Change Background":  # Grayscale functionality
                button.clicked.connect(self.Background_Change)
            elif name == "Meeting Background Change":  # Grayscale functionality
                button.clicked.connect(self.change_back_ground_video)
            elif name == "Face Recognition":  # Grayscale functionality
                button.clicked.connect(self.run_face_recognition)
            elif name == "Face Antisoofing":  # Grayscale functionality
                button.clicked.connect(self.antiSoofing_start)
            elif name == "DMS":  # Grayscale functionality
                button.clicked.connect(self.run_driver_fatigue)
            elif name == "Lane Departure":  # Grayscale functionality
                button.clicked.connect(self.play_laneDeparture_video)
            elif name == "Crowd Count":  # Grayscale functionality
                button.clicked.connect(self.run_crowd_count)  # #
            elif name == "Speech Recognition":  # Grayscale functionality
                button.clicked.connect(self.startVolumeControl)
            elif name == "Exit":  # Grayscale functionality
                button.clicked.connect(self.exit)
            #elif name == "Settings":  # Grayscale functionality
                #button.clicked.connect(self.convert_to_grayscale)

            button.setStyleSheet("""
                QPushButton {
                    color: black; font-size: 16px;  /* Default label color */;
                }
                QPushButton:hover {
                    color: red; /* Label color when mouse is over */
                }
            """)

            self.first_layout.addWidget(button)
            self.first_layout.addStretch()

        # Add first panel to bottom layout
        self.bottom_layout.addWidget(self.first_panel, stretch=1)

        # Second panel: Image panels
        self.second_panel = QFrame()
        self.second_panel.setStyleSheet("background-color: white; border-radius: 10px;")
        self.second_layout = QVBoxLayout(self.second_panel)

        # Application description
        self.app_label = QLabel(" SamSan V-Sense")
        self.app_label.setStyleSheet("font-size: 36px;  color: blue;")
        self.app_label.setAlignment(Qt.AlignCenter)
        self.app_label.setFixedHeight(40)

        # Image container (holds all image panels)
        self.image_container = QFrame()
        self.image_container_layout = QHBoxLayout(self.image_container)

        for i in range(1, 4):  # Create three image panels
            # Shadow effect for each image panel
            shadow_container = QFrame()
            shadow_container.setStyleSheet("""
                background-color: white;
                border-radius: 10px;
                padding: 5px;
                box-shadow: 5px 5px 2px rgba(0, 0, 0, 0.5);
               
                border-radius: 10px;
            """)
            shadow_layout = QVBoxLayout(shadow_container)

            image = QLabel("No Image")
            image.setAlignment(Qt.AlignCenter)
            image.setStyleSheet("background-color: #dddddd; border-radius: 10px; padding: 5px;")
            image.setFixedSize(300, Image_Height)  # Fixed size of image panels
            self.image_labels[f"image_{i}"] = image

            shadow = QGraphicsDropShadowEffect()
            shadow.setOffset(15, 15)  # Shadow offset
            shadow.setBlurRadius(15)  # Shadow blur radius
            shadow.setColor(QColor(0, 0, 0, 150))  # Shadow color (RGBA)
            self.image_labels[f"image_{i}"].setGraphicsEffect(shadow)


            button = QPushButton(f"Upload Image  ")
            button.setStyleSheet("background-color: lightblue; color: black; font-size: 16px;")

            button.setFixedHeight(40)
            if i == 1:  # Upload to image1
                button.clicked.connect(lambda checked, label=image: self.upload_image(label))
            elif i == 2:  # Upload to image2
                button.clicked.connect(lambda checked, label=image: self.upload_image(label))
            elif i == 3:  # Upload to image2
                button.clicked.connect(lambda checked, label=image: self.clear_images(label))
                button.setText("ClearImages")
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)  # Blurring of the shadow
            shadow.setXOffset(5)  # Horizontal offset
            shadow.setYOffset(5)  # Vertical offset
            shadow.setColor(QColor(0, 0, 0, 150))  # Shadow color (black with 150 alpha)

            # Apply the shadow effect to the button
            button.setGraphicsEffect(shadow)

            shadow_layout.addWidget(image)
            shadow_layout.addWidget(button)
            shadow_container.setLayout(shadow_layout)  # Add shadow layout to container
            self.image_container_layout.addWidget(shadow_container)

        self.image_labels["image_1"].setText("Image1")
        self.image_labels["image_2"].setText("Image2")
        self.image_labels["image_3"].setText("Processed Image")

        self.progress_bar = CircularProgress(self.image_labels["image_2"])
        self.progress_bar.setGeometry(70, 100, 150, 150)  # Centered on panel
        self.progress_bar.hide()  #
        self.speech_button = QPushButton("Speak",self.image_labels["image_2"])
        self.speech_button.setStyleSheet("font-size: 16px; color: blue;")
        self.speech_button.setFixedSize(100, 30)
        self.speech_button.setGeometry(90, 300, 210, 323)  # Positioned at the bottom of panel
        self.speech_button.clicked.connect(self.toggleSpeech)
        self.speech_button.hide()  # Hidden initially
        #self.speech_button.setStyleSheet("font-size: 20px ")  # ;font-weight: bold;
        #self.speech_button.setStyleSheet("QPushButton:hover {background-color: red;}")

        self.second_layout.addWidget(self.app_label)
        self.second_layout.addWidget(self.image_container)

        # Add second panel to bottom layout
        self.bottom_layout.addWidget(self.second_panel, stretch=4)

        # Combine bottom layout into the main layout
        self.main_layout.addLayout(self.bottom_layout)

        #window_bg_color = self.palette().window().color().name()
        # background-color: {window_bg_color};  /* Match window color */

    def upload_image(self, label):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.file_path=file_path
            pixmap = QPixmap(file_path).scaled(label.width(), label.height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            if label == self.image_labels["image_1"]:
                self.file_path = file_path
                self.image1 = cv2.imread(file_path)
            elif label == self.image_labels["image_2"]:
                self.file_path = file_path
                self.image2 = cv2.imread(file_path)
            print("Image uploaded successfully.")

    def convert_to_grayscale(self):
        if self.image1:
            original_image = self.image1
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # Convert grayscale image to QPixmap for display
            height, width = gray_image.shape
            bytes_per_line = width
            q_image = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image).scaled(300, Image_Height, Qt.KeepAspectRatioByExpanding)

            # Set the grayscale image in the image3 panel
            self.image_labels["image_3"].setPixmap(pixmap)
            print("Grayscale conversion performed and displayed on Image 3.")

    def perform_edge_detection(self):
        if self.uploaded_image_path:
            original_image = cv2.imread(self.uploaded_image_path)
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)

            height, width = edges.shape
            bytes_per_line = width
            q_image = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image).scaled(300, Image_Height, Qt.KeepAspectRatioByExpanding)
            self.image_labels["image_3"].setPixmap(pixmap)
            print("Edge detection performed and displayed on Image 3.")

    def startVolumeControl(self):

        self.image_labels["image_1"].clear()
        self.image_labels["image_2"].clear()
        self.image_labels["image_3"].clear()
        # Show Circular Progress Bar and Start Speech Button
        self.progress_bar.setProgress(self.current_volume)
        self.progress_bar.show()
        self.speech_button.show()
        self.showMessage("Volume control functionality started.")

    def startVolumeControl1(self):
        self.image_labels["image_2"] ={}
        # Show Circular Progress Bar and Start Speech Button
        self.progress_bar.setProgress(self.current_volume)
        self.progress_bar.show()
        self.speech_button.show()
        self.showMessage("Volume control functionality started.")

    def toggleSpeech(self):
        if self.speech_running:
            self.speech_running = False
            self.speech_button.setText("Start Speech")
            self.progress_bar.hide()
            self.speech_button.hide() # Hide progress bar when speech stops
            self.showMessage("Speech Recognition Stopped!")
        else:
            self.speech_running = True
            self.speech_button.setText("Stop Speech")

            self.progress_bar.show()  # Show progress bar when speech starts
            self.progress_bar.setProgress(self.current_volume)
            self.showMessage("Speech Recognition Started!")
            threading.Thread(target=self.runSpeechRecognition).start()

    def runSpeechRecognition(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            while self.speech_running:
                try:
                    self.showMessage("Listening for commands...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    command = recognizer.recognize_google(audio).lower()

                    if "increase volume" in command:
                        self.increaseVolume()
                    elif "decrease volume" in command:
                        self.decreaseVolume()
                    elif "mute" in command:
                        self.setVolume(0)
                        self.showMessage("Muted the volume.")
                except sr.UnknownValueError:
                    self.showMessage("Could not understand the command.")
                except sr.RequestError as e:
                    self.showMessage(f"Speech recognition error: {e}")
                except Exception as e:
                    self.showMessage(f"Error: {e}")

    def increaseVolume(self):
        # Increase system volume by 10%
        new_volume = min(self.current_volume + 10, 100)
        self.setVolume(new_volume)
        self.showMessage(f"Volume increased to {new_volume}%.")

    def decreaseVolume(self):
        # Decrease system volume by 10%
        new_volume = max(self.current_volume - 10, 0)
        self.setVolume(new_volume)
        self.showMessage(f"Volume decreased to {new_volume}%.")

    def getVolumeControl(self):
        # Initialize the pycaw volume control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return interface.QueryInterface(IAudioEndpointVolume)

    def setVolume(self, volume):
        # Set the system volume and update the progress bar
        self.volume_control.SetMasterVolumeLevelScalar(volume / 100, None)
        self.current_volume = volume
        self.progress_bar.setProgress(volume)

    def showMessage(self, message):
        # Display a simple message
        print(message)

    def load_logo(self, logo_path):
        """Load a logo image and display it in the top-center logo area."""
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaled(
                self.logo_label.width(), self.logo_label.height(), Qt.IgnoreAspectRatio))
        else:
            self.logo_label.setText("Logo")

    def upload_image1(self):
        """Load the first image."""
        if self.speech_running:
            self.speech_running = False  # Stop speech recognition
            self.progress_bar.hide()  # Hide the progress bar
            self.speech_button.hide()  # Hide the Start Speech button
            self.showMessage("Stopped volume control and speech recognition.")

        self.file_name, _ = QFileDialog.getOpenFileName(self, "Select Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if self.file_name:
            self.image1 = cv2.imread(self.file_name)
            self.display_image(self.image_label1, self.image1)
            #cv2.imshow('a',self.image1)
            #cv2.waitKey(0)

    def upload_image2(self):
        """Load the second image."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image2 = cv2.imread(file_name)
            self.display_image(self.image_label2, self.image2)

    def display_image(self, image, label):
            """Display an OpenCV image in a QLabel."""
            if image is None:
                return
            original_height, original_width = self.image1.shape[:2]
            self.image3 = cv2.resize(self.image3, (original_width, original_height),interpolation=cv2.INTER_LINEAR)

            image = cv2.cvtColor(self.image3, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            # self.image_labels["image_3"].setPixmap(pixmap)
            self.image_labels["image_3"].setPixmap(
                pixmap.scaled(self.image_labels["image_3"].width(), self.image_labels["image_3"].height(),
                              Qt.KeepAspectRatio))

    def clear_images(self,label):
        """Function to clear images from all panels"""
        self.image_labels["image_1"].clear()
        self.image_labels["image_2"].clear()
        self.image_labels["image_3"].clear()
        self.image1 = None
        self.image2 = None
        self.image3 = None
        # Reset placeholder text (optional)
        self.image_labels["image_1"].setText("Image1")
        self.image_labels["image_2"].setText("Image2")
        self.image_labels["image_3"].setText("Image3")
        self.speech_button.hide()
        self.progress_bar.hide()

        # Clear the input panel
        #self.input_panel.clear()

    def raise_alert(self,label):
        font = QFont()
        font.setPointSize(28)  # Increase font size to 16 points
        self.image_label1.setFont(font)
        if label=="image1":
            self.image_label1.setText("Load Image1.")
        if label == "image2":
            self.image_label1.setText("Load Image2")

    def convert_to_grayscale1(self):
        """Convert Image 1 to grayscale and display in Panel 4."""
        if self.image1 is None:
            self.output_label.setText("Upload Image 1 first")
            return
        gray_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        self.display_image(self.output_label, gray_image)

    def edge_detection(self):
        """Apply edge detection."""
        if self.image1 is None:
            self.output_label.setText("Upload Image 1 first")
            return
        edges = cv2.Canny(self.image1, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        self.display_image(self.output_label, edges)

    def Background_Change(self):
        if self.image1 is not None and self.image2 is not None:
            image_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
            background = cv2.resize(self.image2, (self.image2.shape[1], self.image2.shape[0]))  # Resize to match input image
            # Apply segmentation
            result = segment.process(image_rgb)
            # Create a mask from segmentation output
            mask = result.segmentation_mask
            mask = (mask > 0.5).astype(np.uint8)  # Thresholding (1 for foreground, 0 for background)
            # Blend the foreground (person/object) with the new background
            output = np.where(mask[:, :, None] == 1, self.image1, background)
            self.image3 = output
            self.display_image(self, self.image3)
        else:
            if self.image1 is not None :
                self.raise_alert("image1")
            if self.image2 is not None :
                self.raise_alert("image2")

    def Remove_Background(self):
        if self.image1 is not None:
            image_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
            # Apply segmentation
            results = segment.process(image_rgb)
            mask = results.segmentation_mask
            threshold = 0.5  # Define threshold for segmentation
            binary_mask = (mask > threshold).astype(np.uint8)
            #binary_mask=cv2.bitwise_not(binary_mask)
            # Convert mask to 3 channels
            #binary_mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])
            # Apply mask to extract foreground
            foreground = cv2.bitwise_and(image_rgb, image_rgb, mask=binary_mask)
            # Create a transparent background (RGBA)
            h, w, _ = image_rgb.shape
            white_background = np.ones((h, w, 3), dtype=np.uint8) * 255  # White RGB values

            # Apply the mask to replace background pixels with white
            white_background[binary_mask == 1] = foreground[binary_mask == 1]
            transparent_background = np.zeros((h, w, 4), dtype=np.uint8)
            transparent_background[:, :, :3] = white_background
            transparent_background[:, :, 3] = binary_mask * 255  # Alpha channel
            self.image3 = transparent_background
            self.display_image(self, self.image3)
        else:
            self.raise_alert("image1")

    def Fog_Remove(self):
        if self.image1 is not None:
            HazeCorrectedImg, haze_map = image_dehazer.remove_haze(self.image1, showHazeTransmissionMap=False)
            self.image3 = HazeCorrectedImg
            self.display_image(self, self.image3)
        else:
            self.raise_alert("image1")

    def run_crowd_count(self):
        if self.image1 is not None:
            img = self.file_path
            count = LWCC.get_count(img, model_name="DM-Count", model_weights="SHB")
            count = int(np.round(count, 0))
            cv2.putText(self.image1, f"Number of Persons: {count}", (100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            self.image3=  self.image1

            self.display_image(self, self.image3)
            # self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
            # h, w, ch = self.image1.shape
            # bytes_per_line = ch * w
            # q_image = QImage(self.image1.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # pixmap = QPixmap.fromImage(q_image)
            # # self.image_labels["image_3"].setPixmap(pixmap)
            # self.image_labels["image_3"].setPixmap(
            #     pixmap.scaled(self.image_labels["image_3"].width(), self.image_labels["image_3"].height(),
            #                   Qt.KeepAspectRatio))

        else:
            print("No Image")
            self.raise_alert("image1")

    def exit(self):
        sys.exit(app.exec_())

    def start_video(self):
        self.image_labels["image_1"].clear()
        self.image_labels["image_2"].clear()
        self.image_labels["image_3"].clear()
        # Start the video capture
        if self.cap:
            self.cap.release()  # Release previous capture if any
        self.cap = cv2.VideoCapture(0)  # Open the default camera
        if not self.cap.isOpened():
            print("Error: Cannot open video source")
            return
        self.timer.start(30)  # Start the timer to update frames

    def update_frame(self):
       #print( self.current_mode)
        #if self.keyboard.is_pressed=='q':
        if self.current_mode == "change_background":
            self.update_change_background_video()

        if self.current_mode == "face_recognition":
            self.update_face_recognition()
        if self.current_mode == "driver_fatigue":
            self.update_drowsiness()
        if self.current_mode == "Lane_departure":
           self.update_lanedeparture()
        if self.current_mode == "AnuSoofing":
           self.run_Anti_Soofing()
        if self.current_mode == "deephoro":
           self.update_deephoro()

    def stop_video(self):
        # Stop the timer and release the video capture
        self.timer.stop()  # Stop updating frames
        if self.cap:
            self.cap.release()  # Release the camera resource

        self.image_label2.clear()  # Clear the QLabel
        self.image_label2.setText("Video Stopped")  # Display placeholder text

    def run_face_recognition(self):
        # Run the Face Recognition app
        self.current_mode = "face_recognition"
        self.start_video()

    def run_driver_fatigue(self):
        # Run the Driver Fatigue Detection app
        self.current_mode = "driver_fatigue"
        self.start_video()

    def change_back_ground_video(self):
        # Run the Driver Fatigue Detection app
        self.current_mode = "change_background"
        self.start_video()

    def antiSoofing_start(self):
        # Run the Driver Fatigue Detection app
        self.current_mode = "AnuSoofing"
        self.start_video()

    def update_face_recognition(self):
        # Update the right video panel
        #print(self.current_mode)
        with open('C:/PythonProject/known_name_encodings', 'rb') as pickle_file:
            known_name_encodings = pickle.load(pickle_file)
        with open('C:/PythonProject/known_names', 'rb') as pickle_file:
            known_names = pickle.load(pickle_file)
            #print(known_names)
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = fr.face_locations(frame)
                face_encodings = fr.face_encodings(frame, face_locations)
                face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    face_distances = fr.face_distance(known_name_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    if confidence > 0.45:
                        name = known_names[best_match_index]
                    else:
                        name = "Unknown"
                    # Convert to QImage
                    #faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, f"{name}", (left + 6, bottom - 6), font, 2.0, (0, 0, 255), 2)
                    height, width, channel = frame.shape
                    #height, width, channel = rgb_frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    self.image_labels["image_2"].setPixmap(
                        pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                      Qt.KeepAspectRatio))
                    left = left - 70
                    right = right + 70
                    top = top - 70
                    bottom = bottom + 70

                    #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cropped_face = frame[top:bottom, left:right]
                    cv2.imwrite("C:/PythonProject/Image_data/deepface_images/sample.jpg", cropped_face)
                    #print("Cropped face saved as cropped_face.jpg")
                    # objs = DeepFace.analyze(
                    #     img_path="C:/PythonProject/Image_data/deepface_images/sample.jpg",
                    #     actions=['age', 'gender', 'race', 'emotion'])
                    #
                    # highest_gender = objs[0]['dominant_gender']
                    # dominant_emotion = objs[0]['dominant_emotion']
                    # dominant_race = objs[0]['dominant_race']
                    # cv2.putText(frame, "Age: {:.2f}".format(objs[0]['age']), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #             (0, 0, 255), 2)
                    # cv2.putText(frame, f"Gender: {highest_gender}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #             (0, 0, 255), 2)
                    #
                    # cv2.putText(frame, f"Race: {dominant_race}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                    #             2)
                    # cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #             (0, 0, 255), 2)
                    #

                    #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    #cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
                    # font = cv2.FONT_HERSHEY_DUPLEX
                    # cv2.putText(frame, f"{name}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    # q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    # self.image_label2.setPixmap(QPixmap.fromImage(q_image))
            else:
                print("Error: Cannot read frame")

    def update_drowsiness(self):
        # Update the right video panel
        alarm_status=false
        alarm_status2=false
        def alarm(msg):
            global alarm_status
            global alarm_status2
            global saying

            # while alarm_status:
            #     print('Closed eyes call')
            #     saying = True
            #     sound1.play()
            #     saying = false
            # if alarm_status2:
            #     print('Yawn call')
            #     sound2.play()

        def eye_aspect_ratio(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            ear = (A + B) / (2.0 * C)
            return ear

        def final_ear(shape):
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            return (ear, leftEye, rightEye)

        def lip_distance(shape):
            top_lip = shape[50:53]
            top_lip = np.concatenate((top_lip, shape[61:64]))
            low_lip = shape[56:59]
            low_lip = np.concatenate((low_lip, shape[65:68]))
            top_mean = np.mean(top_lip, axis=0)
            low_mean = np.mean(low_lip, axis=0)
            distance = abs(top_mean[1] - low_mean[1])
            return distance

        ap = argparse.ArgumentParser()
        ap.add_argument("-w", "--webcam", type=int, default=0,
                        help="index of webcam on system")
        args = vars(ap.parse_args())

        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 35
        YAWN_THRESH = 15  # 30
        YAWN_CONSEC_FRAMES = 15 #30  # New threshold for yawn detection
        alarm_status = false
        alarm_status2 = false
        saying = false
        COUNTER = 0
        YARN_FRAME = 0  # New counter for yawn frames
        print("-> Loading the predictor and detector...")
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        print("-> Starting Video Stream")
        #self.cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.cap.read(0)
            #frame = imutils.resize(self.frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = final_ear(shape)
                ear = eye[0]
                leftEye = eye[1]
                rightEye = eye[2]

                distance = lip_distance(shape)
                #(distance)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                #cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
                #print("ear==",ear)
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_status:
                            alarm_status = True
                            t = Thread(target=alarm, args=('wake up sir',))
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "DROWSINESS ALERT!", (225, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_status = False

                if distance > YAWN_THRESH:
                    YARN_FRAME += 1

                    if YARN_FRAME >= YAWN_CONSEC_FRAMES:
                        if not alarm_status2:
                            alarm_status2 = True
                            t = Thread(target=alarm, args=('take some fresh air sir',))
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "Yawn Alert", (225, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    YARN_FRAME = 0
                    alarm_status2 = False

                if ear<0.10:
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (225, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if distance > 17 :
                    cv2.putText(frame, "YAWN: {:.2f}".format(distance), (225, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape

                font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(cv2.imshow('Driver Identification', frame)
                #             if cv2.waitKey(1) & 0xFF == ord('q'):
                #                 Driver_id = 0
                #                 cv2.destroyAllWindows()
                #                 break, f"{name} - {confidence:.2f}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                #cv2.imshow('Driver Identification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    #Driver_id = 0
                    cv2.destroyAllWindows()
                    break
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_labels["image_2"].setPixmap(
                    pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                  Qt.KeepAspectRatio))

    def play_laneDeparture_video(self):
        """Start playing the recorded video frame by frame using QTimer."""
        if not self.cap_recorded:
            # Initialize the video capture object
            self.cap_recorded = cv2.VideoCapture('C:/PythonProject/VID_20230906_174106981_Trim.mp4')

            # Check if the video file opened successfully
            if not self.cap_recorded.isOpened():
                print("Error: Unable to open the video file.")
                self.cap_recorded = None
                return

            # Start the QTimer to call `update_laneDeparture_video` every 30ms
            self.timer_recorded.start(30)

    def update_laneDeparture_video(self):
        if self.cap_recorded:
            run=0
            ret, image = self.cap_recorded.read()
            # Restart video if it ends
            if not ret:
                print("Restarting video...")
                self.cap_recorded.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
                return  # Exit to avoid processing invalid frames
            try:
                # Process the frame (example: convert to grayscale)
                image = cv2.resize(image, (700, 900))
                # Step 2: Reading the Image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Step 3: Converting to Grayscale
                blur = cv2.GaussianBlur(gray, (5, 5), 0)

                # Step 4: Gaussian Blur
                # edges = cv2.Canny(blur, 50, 150)  # working
                edges = cv2.Canny(blur, 40, 165)

                # Step 5: Canny Edge Detection
                height, width = image.shape[:2]
                # Define percentages for ROI (adjust these as needed)
                top_left = (0, int(height * 0.8))
                top_right = (width - 1, int(height * 0.8))
                bottom_left = (0, height - 1)
                bottom_right = (width - 1, height - 1)

                roi_vertices = [top_left, top_right, bottom_right, bottom_left]

                # roi_vertices = [(0, height/1.9), (width-width*.5, height/3), (width-width*0.2, height-height*.6)]  # working to some extent
                # roi_vertices = [(0, height / 2.4), (width / 1.5, height / 3.2),
                #                (width - width * 0.8, height - height * .2)]
                mask_color = 255
                mask = np.zeros_like(edges)

                cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)

                masked_edges = cv2.bitwise_and(edges, mask)

                # Step 6: Region of Interest
                # lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25) # original
                # lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi / 60, threshold=100, minLineLength=25, maxLineGap=30) # working to some extent
                lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi / 60, threshold=90, minLineLength=25,
                                        maxLineGap=45)
                # Step 7: Hough Transform
                line_image = np.zeros_like(image)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  original
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # Step 8: Drawing the Lines
                # final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0) original
                final_image = cv2.addWeighted(image, .9, line_image, 2, 1)

                height, width, channel = final_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(final_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                key = cv2.waitKey(1)  # Delay allows interaction
                if key == 27:  # ESC key
                    print("ESC pressed. Stopping video playback.")
                    self.stop_laneDeparture_video()
                self.image_labels["image_2"].setPixmap(
                    pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                  Qt.KeepAspectRatio))

                # Check for ESC key press using OpenCV
            except:
                self.timer_recorded.stop()
                self.cap_recorded.release()
                self.cap_recorded = None
                self.image_label2.setText("Recorded Video playback finished.")
                self.stop_laneDeparture_video()

    def stop_laneDeparture_video(self):
        """Stop video playback and release resources."""
        if self.timer_recorded.isActive():
            self.timer_recorded.stop()  # Stop the QTimer
        if self.cap_recorded:
            self.cap_recorded.release()  # Release the video capture object
            self.cap_recorded = None
        self.video_label.setText("Video stopped. Press play to restart.")  # Update UI

    def detect_vehicles(frame, yolo_model):
        """
        Detect vehicles using YOLO model and return bounding boxes.
        """
        results = yolo_model.predict(frame)
        detected_boxes = []
        for result in results:
            for box in result.boxes.xyxy:  # Extract bounding box coordinates
                detected_boxes.append(box.tolist())
        return detected_boxes

    def create_vehicle_mask(frame, detected_boxes):
        """
        Create a mask for regions where vehicles are detected.
        """
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for box in detected_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill vehicle bounding boxes
        return mask


    def update_lanedepartureq111(self):
        #print("again",self.current_mode)
        breakk= 0
        while (breakk== 0):
            isclosed = 0
            countframe = 0
            missed = 0
            video_capture = cv2.VideoCapture('C:/PythonProject/Image_data/my_Car_lane_detection.mp4')
            # Check if 'q' key is pressed
            while (breakk== 0):
                ret, image = video_capture.read()
                # image = cv2.resize(image, (480, 860))
                if (ret == True):
                    image = cv2.resize(image, (700, 900))
                    # Step 2: Reading the Image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Step 3: Converting to Grayscale
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)

                    # Step 4: Gaussian Blur
                    # edges = cv2.Canny(blur, 50, 150)  # working
                    edges = cv2.Canny(blur, 40, 165)

                    # Step 5: Canny Edge Detection
                    height, width = image.shape[:2]
                    #   roi_vertices = [(0, height/1.9), (width-width*.5, height/3), (width-width*0.2, height-height*.6)]  # working to some extent
                    roi_vertices = [(0, height / 2.4), (width / 1.5, height / 3.2),
                                    (width - width * 0.8, height - height * .2)]
                    mask_color = 255
                    mask = np.zeros_like(edges)
                    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
                    masked_edges = cv2.bitwise_and(edges, mask)

                    # Step 6: Region of Interest
                    # lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25) # original
                    # lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi / 60, threshold=100, minLineLength=25, maxLineGap=30) # working to some extent
                    lines = cv2.HoughLinesP(masked_edges, rho=10, theta=np.pi / 60, threshold=90, minLineLength=25,
                                            maxLineGap=45)
                    # Step 7: Hough Transform
                    line_image = np.zeros_like(image)
                    try:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  original
                            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                        # Step 8: Drawing the Lines
                        # final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0) original
                        final_image = cv2.addWeighted(image, .9, line_image, 2, 1)
                        #cv2.imshow('Lane Departure Warning', final_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            1

                        # Step 9: Overlaying the Lines on the Original Image
                        countframe = countframe + 1
                        height, width, channel = final_image.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(final_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        self.image_labels["image_2"].setPixmap(
                            pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                          Qt.KeepAspectRatio))
                        #self.image_label2.setPixmap(QPixmap.fromImage(q_image))
                        if keyboard.is_pressed('q'):
                            print("Exiting loop!")
                            breakk= 1

                        else:
                            print("Waiting for 'q' key...")
                    except:
                        missed = missed + 1
                        countframe = countframe + 1
                        # print(countframe)
                else:
                    break
            if keyboard.is_pressed('q'):
                self.image_label2.clear()
                break
        #if x1 > width / 2 - width * .05:
            #cv2.putText(final_image, "You are Departing Lane", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def detect_and_match_features(self):
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(self.image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(self.image2, None)
        # keypoints3, descriptors3 = orb.detectAndCompute(img3, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return keypoints1, keypoints2, matches

    def estimate_homography(self,keypoints1,keypoints2, matches):
        threshold = 3
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
        return H, mask

    def warp_images(self, H):
        h1, w1 = self.image2.shape[:2]
        h2, w2 = self.image1.shape[:2]
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)
        corners = np.concatenate((corners1, warped_corners2), axis=0)
        [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        warped_img2 = cv2.warpPerspective(self.image1, Ht @ H, (xmax - xmin, ymax - ymin))
        warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = self.image2
        return warped_img2

    def blend_images(self,warped_img,imgg1):
        mask = np.where(warped_img != 0, 1, 0).astype(np.float32)
        #print("mask2=", mask.shape)
        #print(mask.shape)
        blended_img = warped_img * mask + imgg1 * (1 - mask)
        return blended_img.astype(np.uint8)

    def stitch_images(self):
        if self.image1 is None or self.image2 is None:
            self.output_label.setText("Upload both images first")
            return
        keypoints1, keypoints2, matches = self.detect_and_match_features()
        H, mask = self.estimate_homography(keypoints1,keypoints2,matches)
        warped_img = self.warp_images(H)
        img1 = cv2.resize(self.image1, (warped_img.shape[1], warped_img.shape[0]))
        output_img = self.blend_images(warped_img,img1)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        self.image3=output_img
        self.display_image(self, self.image3)

    def closeEvent(self, event):
        # Release resources when the window is closed
        if self.cap:
            self.cap.release()
        super().closeEvent(event)

    def update_change_background_video(self):
        #update_change_background
        img2 = self.image2
        if self.cap:
            ret, img = self.cap.read()
            if ret:
                segmentor = SelfiSegmentation()
                #imgBg = cv2.imread("C:/PythonProject/Image_data/ImageStitching/img1/L1.jpg")

                hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

                # Define color range for green
                lower_green = np.array([35, 55, 55])
                upper_green = np.array([85, 255, 255])

                # Create a mask
                mask = cv2.inRange(hsv, lower_green, upper_green)

                # Smooth edges using Gaussian Blur
                mask = cv2.GaussianBlur(mask, (15, 15), 0)

                # Apply mask to extract the subject
                img2 = cv2.bitwise_and(img2, img2, mask=~mask)
                img = cv2.resize(img, (470, 540))
                """Process and update each frame of the recorded video in the QLabel."""
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                backimage = cv2.resize(img2, (470, 540))
                new_image = segmentor.removeBG(img, backimage)
                height, width, channel = new_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(new_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_labels["image_2"].setPixmap(
                    pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                  Qt.KeepAspectRatio))


    def run_Anti_Soofing(self):
        model_dir = "C:/PythonProject/resources/anti_spoof_models/"

        desc = "test"
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument(
            "--model_dir",
            type=str,
            # default="C:/PythonProject/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/",
            default=model_dir,
            help="model_lib used to test")
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()
        #cap = cv2.VideoCapture(0)
        #while True:
        if self.cap:
            ret, image = self.cap.read()
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if ret:
                height, width, channel = image.shape
                height1 = int(width * 3 / 4)
                image1 = cv2.resize(image, (width, height1))
                # image = cv2.imread(SAMPLE_IMAGE_PATH + "Pradeep.jpg")
                # result = check_image(image1)
                image_bbox = model_test.get_bbox(image)
                prediction = np.zeros((1, 3))
                test_speed = 0
                # sum the prediction from single model's result
                for model_name in os.listdir(model_dir):
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": image,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img = image_cropper.crop(**param)
                    start = time.time()
                    prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                    # test_speed += time.time() - start
                # draw result of prediction
                label = np.argmax(prediction)
                value = prediction[0][label] / 2
                if label == 1:
                    # print("Image '{}' is Real Face. Score: {:.2f}.".format("Pradeep", value))
                    # result_text = "RealFace Score: {:.2f}".format(value)
                    result_text = "RealFace"
                    color = (0, 255, 0)
                else:
                    # print("Image '{}' is Fake Face. Score: {:.2f}.".format("Pradeep", value))
                    # result_text = "FakeFace Score: {:.2f}".format(value)
                    result_text = "FakeFace"
                    color = (255, 0, 0)
                # print("Prediction cost {:.2f} s".format(test_speed))
                # cv2.rectangle(
                #     image,
                #     (image_bbox[0], image_bbox[1]),
                #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                #     color, 2)
                cv2.putText(
                    image,
                    result_text,
                    (image_bbox[0]+20, image_bbox[1] - 50),
                    cv2.FONT_HERSHEY_COMPLEX, 2 * image.shape[0] / 1024, color,2)

                with open('C:/PythonProject/known_name_encodings', 'rb') as pickle_file:
                    known_name_encodings = pickle.load(pickle_file)
                with open('C:/PythonProject/known_names', 'rb') as pickle_file:
                    known_names = pickle.load(pickle_file)
                    # print(known_names)

                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = fr.face_locations(frame)
                face_encodings = fr.face_encodings(frame, face_locations)
                face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    face_distances = fr.face_distance(known_name_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    if confidence > 0.45:
                        name = known_names[best_match_index]
                    else:
                        name = "Unknown"
                    # Convert to QImage
                    # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image, f"{name}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)

                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_labels["image_2"].setPixmap(
                    pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                  Qt.KeepAspectRatio))
                #cv2.imshow("Face Detection", image)
                #     # Break the loop on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    1

    def keyPressEvent(self, event: QKeyEvent):
        """Stop both live and recorded videos on key press."""

        if event.key() == Qt.Key_Escape:
            try:
                self.stop_laneDeparture_video()
            except:
                1
        if event.key() == Qt.Key_S:  # Stop when 'S' key is pressed
            # Stop live video
            try:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                    self.image_label2.clear()
                    self.image2 = None
                    self.image_label2.setText("Live Video stopped.")
                    self.clear_images(self, label)
            except:
                1


                # Stop recorded video
                try:
                    if self.timer_recorded.isActive():
                        self.timer_recorded.stop()
                    if self.cap_recorded:
                        self.cap_recorded.release()
                        self.cap_recorded = None
                    self.image_label2.clear()
                    self.image_label2.setText("Recorded Video stopped.")
                except:
                    1



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    window.setWindowTitle('SAMSAN V-SENSE')
    window.setStyleSheet("background-color: white;")
    window.move(5, 5)
    window.setFixedSize(1360, 650)
    sys.exit(app.exec_())
