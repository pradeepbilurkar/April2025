import warnings
warnings.filterwarnings("ignore")
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QFrame, QMainWindow,QMenuBar,QAction, QGraphicsDropShadowEffect
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
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
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
from ultralytics import YOLO
from cv2 import Stitcher_create
import subprocess

from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from queue import Queue
import pickle

from skimage.morphology import disk
from skimage.measure import label, shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, regionprops_table
from skimage import img_as_ubyte


model = YOLO("yolov8n.pt")

mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')
Image_Height = 350

class ClickableLabel(QLabel):
    """ Custom QLabel that detects mouse clicks """

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        """ Open file dialog when clicked """
        parent = self.parent()
        if parent:
            parent.select_images()

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


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Initialize UI and Layouts
        self.init_ui()

        # Create menu bar
        self.create_menu()


    def init_ui(self):
        # Set up the window
        self.setWindowTitle("SAMSAN V-SENSE")
        self.setStyleSheet("background-color: white;")
        self.setFixedSize(1360, 650)

        # Top Layout: Logo
        self.top_layout = QHBoxLayout()
        self.logo = QLabel()
        self.logo_pixmap = QPixmap("C:/Users/Samsan/Desktop/logo.png")  # Replace with your logo path
        self.logo.setPixmap(self.logo_pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignLeft)
        self.top_layout.addWidget(self.logo)

        self.main_layout.addLayout(self.top_layout)

        # Bottom Layout
        self.bottom_layout = QHBoxLayout()

        # Image Panels (Using your provided code for upload and display functionality)
        self.second_panel = QFrame()
        self.second_panel.setStyleSheet("background-color: white; border-radius: 10px;")
        self.second_layout = QVBoxLayout(self.second_panel)

        self.app_label = QLabel("SamSan V-Sense")
        self.app_label.setStyleSheet("font-size: 36px; color: blue;")
        self.app_label.setAlignment(Qt.AlignCenter)
        self.app_label.setFixedHeight(40)

        self.image_container = QFrame()
        self.image_container_layout = QHBoxLayout(self.image_container)

        # Using your provided code to create image panels with upload buttons
        self.image_labels = {}
        Image_Height = 300  # Assuming fixed height for image panels

        #self.first_layout.addStretch()

        # Initialize paths and labels for uploaded images
        self.image1 = None  # For image1
        self.image2 = None
        self.image3 = None  # For image2
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



        for i in range(1, 4):  # Create three image panels
            shadow_container = QFrame()
            shadow_container.setStyleSheet(
                "background-color: white; border-radius: 10px; padding: 5px;"
                "box-shadow: 15px 15px 12px rgba(0, 0, 0, 0.5);"
            )
            shadow_layout = QVBoxLayout(shadow_container)

            image = QLabel("No Image")
            image.setAlignment(Qt.AlignCenter)
            image.setStyleSheet("background-color: #dddddd; border-radius: 10px; padding: 5px;")
            image.setFixedSize(400, 350)  # Assuming fixed height of 300
            self.image_labels[f"image_{i}"] = image

            shadow = QGraphicsDropShadowEffect()
            shadow.setOffset(15, 15)  # Shadow offset
            shadow.setBlurRadius(15)  # Shadow blur radius
            shadow.setColor(QColor(0, 0, 0, 150))  # Shadow color (RGBA)
            self.image_labels[f"image_{i}"].setGraphicsEffect(shadow)



            button = QPushButton(f"Upload Image")
            button.setStyleSheet("background-color: lightblue; color: black; font-size: 16px;")
            button.setFixedHeight(40)
            #button.setFixedWidth(120)

            if i == 3:  # Add "Clear Images" functionality for the third button
                button.clicked.connect(lambda checked, label=image: self.upload_image(label))
                button.setText("Upload Image")
            else:  # Upload functionality for the other buttons
                button.clicked.connect(lambda checked, label=image: self.upload_image(label))


            shadow_layout.addWidget(image)
            shadow_layout.addWidget(button)
            shadow_container.setLayout(shadow_layout)
            self.image_container_layout.addWidget(shadow_container)

        self.progress_bar = CircularProgress(self.image_labels["image_2"])
        self.progress_bar.setGeometry(70, 100, 150, 150)  # Centered on panel
        self.progress_bar.hide()  #
        self.speech_button = QPushButton("Speak", self.image_labels["image_2"])
        self.speech_button.setStyleSheet("font-size: 16px; color: blue;")
        self.speech_button.setFixedSize(100, 30)
        self.speech_button.setGeometry(90, 300, 210, 323)  # Positioned at the bottom of panel
        self.speech_button.clicked.connect(self.toggleSpeech)
        self.speech_button.hide()  # Hidden initially



        self.second_layout.addWidget(self.app_label)
        self.second_layout.addWidget(self.image_container)

        self.bottom_layout.addWidget(self.second_panel, stretch=4)

        self.main_layout.addLayout(self.bottom_layout)

    def create_menu(self):
        # Create the menu bar
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        menu_bar.setStyleSheet("""
                QMenuBar::item { 
                    padding: 5px; 
                }
                QMenuBar::item:selected { 
                    color: blue; /* Change main menu text color to blue on hover */ 
                }
                QMenu::item { 
                    padding: 5px; 
                }
                QMenu::item:selected { 
                    color: blue; /* Change submenu text color to blue on hover */
                }
                QMenuBar { 
            font-size: 16px;  /* Increase main menu font size */
        }
        QMenu { 
            font-size: 14px;  /* Increase submenu font size */
        }
        QMenuBar::item:selected { 
            color: blue; /* Change main menu text color to blue on hover */
        }
        QMenu::item:selected { 
            color: blue; /* Change submenu text color to blue on hover */
        }

            """)

        # File Menu
        file_menu = menu_bar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Image Processing Menu
        image_menu = menu_bar.addMenu("Image Processing")

        # Style Transfer Submenu
        style_transfer_menu = image_menu.addMenu("Style Transfer")

        style_models = {
            "Starry Night": "starry_night.t7",
            "The Scream": "the_wave.t7",
            "La Muse": "la_muse.t7"
        }

        for name, model_file in style_models.items():
            action = QAction(name, self)
            action.triggered.connect(lambda checked, file=model_file: self.apply_style_transfer(file))
            style_transfer_menu.addAction(action)

        image_actions = {
            "Edge Detection": self.edge_detection,  # Newly added
            "Convert_to_grayscale":self.convert_to_grayscale,
            "Stitch Image": self.stitch_images,
            "Remove Fog": self.Fog_Remove,
            "Remove Background": self.Remove_Background,
            "Change Background": self.Background_Change,
            "GreenScrenEffect":self.green_Screen_Effect,
            "Super Resolution":self.super_resolution,
            "Match Style": self.histogram_matching,

            #"Style_transfer":self.Style_transfer,
            "Clear Image": self.clear_images
        }
        for name, method in image_actions.items():
            action = QAction(name, self)
            action.triggered.connect(method)
            image_menu.addAction(action)

        # Video Processing Menu
        video_menu = menu_bar.addMenu("Video Processing")
        video_actions = {
            "Meeting Background Change": self.change_back_ground_video,
            "Lane Departure": self.play_laneDeparture_video,
        }
        for name, method in video_actions.items():
            action = QAction(name, self)
            action.triggered.connect(method)
            video_menu.addAction(action)

        # Face Analysis Menu
        face_menu = menu_bar.addMenu("Face Analysis")
        face_actions = {
            "Face Recognition": self.run_face_recognition,
            "Face Antisoofing": self.antiSoofing_start,
        }
        for name, method in face_actions.items():
            action = QAction(name, self)
            action.triggered.connect(method)
            face_menu.addAction(action)

        # Safety & Monitoring Menu
        safety_menu = menu_bar.addMenu("Safety & Monitoring")
        safety_actions = {
            "DMS": self.run_driver_fatigue,
            "Crowd Count": self.run_crowd_count,
        }

        for name, method in safety_actions.items():
            action = QAction(name, self)
            action.triggered.connect(method)
            safety_menu.addAction(action)

            # Safety & Monitoring Menu
        real_manu = menu_bar.addMenu("Real World Applications")
        real_actions = {
            "Node_Detection": self.mammogram,
            "Tyre Quality Mark": self.tyre_mark_detection,
            #"Tyre Mark Matlab": self.run_matlab_app,
        }
        for name, method in real_actions.items():
            action = QAction(name, self)
            action.triggered.connect(method)
            real_manu.addAction(action)

        # real_manu = menu_bar.addMenu("Real World Applications")
        # real_actions = {
        #     "Node_Detection": self.mammogram,
        #     "Tyre Quality Mark": self.tyre_mark_detection,
        #     "Tyre Mark Matlab": run_matlab_app,  # Call MATLAB script
        # }
        #
        # for name, method in real_actions.items():
        #     action = QAction(name, self)
        #     action.triggered.connect(method)
        #     real_manu.addAction(action)

        # Miscellaneous Menu
        misc_menu = menu_bar.addMenu("Miscellaneous")
        misc_actions = {
            "Speech Recognition": self.startVolumeControl,
            "In Paint": self.in_paint_image,
            "Object Detection": self.live_objectDetection_start,
        }
        for name, method in misc_actions.items():
            action = QAction(name, self)
            action.triggered.connect(method)
            misc_menu.addAction(action)

    # Placeholder methods for functionalities
    def upload_image(self, label):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path).scaled(label.width(), label.height(), Qt.KeepAspectRatioByExpanding,
                                               Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            if label == self.image_labels["image_1"]:
                self.file_path = file_path
                self.image1 = cv2.imread(file_path)
            elif label == self.image_labels["image_2"]:
                self.file_path = file_path
                self.image2 = cv2.imread(file_path)
            print("Image uploaded successfully.")

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

    def clear_images_original(self,label):
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


    def clear_images(self, label):
        """Function to clear images from all panels."""

        # Iterate through image labels and reset
        for key in ["image_1", "image_2", "image_3"]:
            self.image_labels[key].clear()
            self.image_labels[key].setText(key.replace("_", " ").title())  # Auto-format placeholder

        # Reset image variables
        self.image1 = self.image2 = self.image3 = None

        # Hide unnecessary UI elements
        #self.speech_button.hide()
        #self.progress_bar.hide()

        # Clear the input panel (if applicable)
        if hasattr(self, "input_panel"):  # Avoid errors if input_panel isn't defined
            self.input_panel.clear()


    def raise_alert(self,label):
        font = QFont()
        font.setPointSize(28)  # Increase font size to 16 points
        self.image_label1.setFont(font)
        if label=="image1":
            self.image_label1.setText("Load Image1.")
        if label == "image2":
            self.image_label1.setText("Load Image2")

    def run_matlab_app(self):
        subprocess.run(["matlab", "-r", r"c:\Matlab Project\TISI_Mark_Detection.mlapp"], shell=True)





    def apply_style_transfer(self, model_file):
        # Load the pre-trained neural style transfer model
        #style_model = ("C:/Users/Samsan/Downloads/"
         #              "style-transfer-models-master/style-transfer-models-master/models/eccv16/starry_night.t7")  # Choose a model (e.g., "mosaic.t7", "starry_night.t7", "the_scream.t7")


        path="C:/PythonProject/style-transfer-models-master/models/eccv16/"
        #modelpath=path+model_file
        style_model =  model_file
        style_model= path+model_file
        net = cv2.dnn.readNetFromTorch(style_model)

        # Load the input image
        image = self.image1

        # Resize image to match model requirements
        (h, w) = image.shape[:2]
        new_width = 600
        new_height = int((h / w) * new_width)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert image to blob format for neural network processing
        blob = cv2.dnn.blobFromImage(resized_image, mean=(103.939, 116.779, 123.680), swapRB=False, crop=False)

        # Apply style transfer
        net.setInput(blob)
        styled_image = net.forward()

        # Reshape and post-process output
        styled_image = styled_image.reshape(3, styled_image.shape[2], styled_image.shape[3])
        styled_image[0] += 103.939
        styled_image[1] += 116.779
        styled_image[2] += 123.680
        styled_image = styled_image.transpose(1, 2, 0)
        styled_image = cv2.convertScaleAbs(styled_image)
        self.image3= styled_image
        self.display_image(self, styled_image)
        # Save and display output
        # cv2.imwrite("styled_output.jpg", styled_image)
        # cv2.imshow("Style Transfer", styled_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def super_resolution(self):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # Choose a model (e.g., "EDSR", "FSRCNN", "ESPCN") with upscale factor
        model_path = "C:/PythonProject/EDSR_x4.pb"  # Example: EDSR model (ensure the file exists)
        sr.readModel(model_path)
        sr.setModel("edsr", 4)  # "edsr" model with 4x upscaling

        # Load the low-resolution image
        image = self.image1

        # Apply Super Resolution
        high_res_image = sr.upsample(image)
        self.image3=high_res_image
        self.display_image(self,high_res_image )
        # Save and display results
        # cv2.imwrite("high_res_output.jpg", high_res_image)
        # cv2.imshow("Super Resolution", high_res_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def green_Screen_Effect(self):
        image = self.image1  # Input image with a green screen
        background = self.image2

        # Resize background to match input image dimensions
        background = cv2.resize(background, (image.shape[1], image.shape[0]))

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the green color range (adjust based on shade)
        lower_green = np.array([35, 50, 50])  # Lower boundary of green color in HSV
        upper_green = np.array([85, 255, 255])  # Upper boundary of green color in HSV

        # Create a mask for detecting green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)  # Inverted mask for extracting person

        # Extract the foreground (person) and background separately
        foreground = cv2.bitwise_and(image, image, mask=mask_inv)
        background_part = cv2.bitwise_and(background, background, mask=mask)

        # Combine foreground and new background
        final_image = cv2.add(foreground, background_part)
        self.image3=final_image
        self.display_image(self,self.image3)
        # Show the final image
        #cv2.imshow("Green Screen Effect", final_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def convert_to_grayscale1(self):
        """Convert Image 1 to gray_imagegrayscale and display in Panel 4."""
        if self.image1 is None:
            self.output_label.setText("Upload Image 1 first")
            return
        gray_image = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        self.image3=gray_image
        self.display_image(self, gray_image)

    def edge_detection(self):
        """Apply edge detection."""
        if self.image1 is None:
            self.output_label.setText("Upload Image 1 first")
            return
        edges = cv2.Canny(self.image1, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        self.image3=edges
        self.display_image(self, edges)

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




    def Background_Change_dynamic(self):
        bg1 = "C:/PythonProject/Image_data/Remove background/img2.png"
        bg2 = "C:/PythonProject/Image_data/Remove background/bgimage.jpg"

        backgrounds = [bg1, bg2, ]  # Add more backgrounds
        background_index = 0  # Start with the first background

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Dynamically load the selected background
            background = cv2.imread(backgrounds[background_index])
            if background is None:
                print(f"Error loading {backgrounds[background_index]}, using default black background.")
                background = cv2.resize(frame, (frame.shape[1], frame.shape[0]))  # Black background alternative

            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

            # Convert frame to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = segment.process(frame_rgb)

            # Create mask for segmentation
            mask = results.segmentation_mask
            mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

            mask_inv = 1 - mask
            foreground = cv2.bitwise_and(frame, frame, mask=mask.astype("uint8"))
            background_part = cv2.bitwise_and(background, background, mask=mask_inv.astype("uint8"))
            output = cv2.add(foreground, background_part)

            cv2.imshow("AI Dynamic Background", output)

            # Change background dynamically using keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Press 'n' to switch to the next background
                background_index = (background_index + 1) % len(backgrounds)
            elif key == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

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
        if self.current_mode == "objectDetection":
           self.update_object_detection_video()


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

    def live_objectDetection_start(self):
        self.current_mode = "objectDetection"
        self.start_video()

    def update_face_recognition(self):
        # Update the right video panel
        #print(self.current_mode)
        path = "C:/PythonProject/Image_data/FaceRecognition/"
        known_names = []
        known_name_encodings = []
        images = os.listdir(path)
        # video_capture = cv2.VideoCapture(0)
        frame_queue = Queue()
        processed_frame_counter = 0
        print_interval = 10
        display_interval = 5
        # for _ in images:
        #     image = fr.load_image_file(os.path.join(path, _))
        #     encoding = fr.face_encodings(image)[0]
        #     known_name_encodings.append(encoding)
        #     known_names.append(os.path.splitext(os.path.basename(_))[0].capitalize())
        #     with open('C:/PythonProject/known_name_encodings', 'wb') as pickle_file:
        #         pickle.dump(known_name_encodings, pickle_file)
        #     with open('C:/PythonProject/known_names', 'wb') as pickle_file:
        #       pickle.dump(known_names, pickle_file)

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
                        if confidence <= 0.45:  # If confidence is low, face is "Unknown"
                           # Define the desired save path
                            save_path = "C:/PythonProject/Image_data/FaceRecognition/"
                            #os.makedirs(path, exist_ok=True)  # Ensure the directory exists

                            # Adjust bounding box safely
                            left = max(left - 70, 0)
                            right = min(right + 70, frame.shape[1])
                            top = max(top - 70, 0)
                            bottom = min(bottom + 70, frame.shape[0])

                            cropped_face = frame[top:bottom, left:right]
                            file_name = f"{save_path}{int(time.time())}.jpg"  # Unique filename using timestamp
                            cv2.imwrite(file_name, cropped_face)

                            # Release video capture and exit
                            self.cap.release()
                            cv2.destroyAllWindows()
                            self.clear_images()
                            print(f"Unrecognized face saved: {file_name}")
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
                    # left = left - 70
                    # right = right + 70
                    # top = top - 70
                    # bottom = bottom + 70
                    #
                    # #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # cropped_face = frame[top:bottom, left:right]
                    # cv2.imwrite("C:/PythonProject/Image_data/deepface_images/sample.jpg", cropped_face)

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

        EYE_AR_THRESH = 0.2   #0.2
        EYE_AR_CONSEC_FRAMES = 35  # 35
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
            text=''
            #frame = imutils.resize(self.frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                nose_x = 300
                shape = predictor(gray, rect)
                #nose_x = shape.part(30).x
                #nose_y = shape.part(30).y
                # text = ""
                # print("Looking:" ,nose_x - 300)
                # if nose_x - 300 > 60:
                #     text = "Looking Left"
                # if nose_x - 300 < -30:
                #     text = "Looking Right"
                nose_x =300
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
                        text = "DROWSINESS ALERT!"
                        print ("DROWSINESS Alert:" ,text)
                        #cv2.putText(frame, text, (75, 300),
                                    #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
                        text = "Yawn Alert"
                        cv2.putText(frame, text, (75, 250),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (155, 0, 0), 2)
                        cv2.imshow('Driver Identification', frame)
                else:
                    YARN_FRAME = 0
                    alarm_status2 = False

                # if ear<0.10:
                #     cv2.putText(frame, "EAR: {:.2f}".format(ear), (225, 60),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # if distance > 17 :
                #     cv2.putText(frame, "YAWN: {:.2f}".format(distance), (225, 70),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.putText(frame, text, (75, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

                text =''
                # cv2.putText(cv2.imshow('Driver Identification', frame)
                #             if cv2.waitKey(1) & 0xFF == ord('q'):
                #                 Driver_id = 0
                #                 cv2.destroyAllWindows()
                #                 break, f"{name} - {confidence:.2f}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.imshow('Driver Identification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                     #Driver_id = 0
                     cv2.destroyAllWindows()
                     break
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                # self.image_labels["image_2"].setPixmap(
                #     pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                #                   Qt.KeepAspectRatio))

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
        #print("input image", self.image1.shape)
        keypoints1, keypoints2, matches = self.detect_and_match_features()
        H, mask = self.estimate_homography(keypoints1,keypoints2,matches)
        warped_img = self.warp_images(H)
        #img1 = cv2.resize(self.image1, (warped_img.shape[1], warped_img.shape[0]))
        # output_img = self.blend_images(warped_img,img1)
        warped_img = cv2.resize(warped_img, (self.image1.shape[1], self.image1.shape[0]))
        output_img = self.blend_images(warped_img,self.image1)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        #print('output image', output_img.shape)
        self.image3=output_img
        self.display_image(self, self.image3)
        cv2.imshow('Output',output_img)
        cv2.waitKey()

    def stitch_images1(self):
        if self.image1 is None or self.image2 is None:
            self.output_label.setText("Upload both images first")
            return
        # print("input image", self.image1.shape)
        keypoints1, keypoints2, matches = self.detect_and_match_features()
        H, mask = self.estimate_homography(keypoints1, keypoints2, matches)
        warped_img = self.warp_images(H)
        # img1 = cv2.resize(self.image1, (warped_img.shape[1], warped_img.shape[0]))
        # output_img = self.blend_images(warped_img,img1)
        warped_img = cv2.resize(warped_img, (self.image1.shape[1], self.image1.shape[0]))
        output_img = self.blend_images(warped_img, self.image1)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        # print('output image', output_img.shape)
        self.image3 = output_img
        self.display_image(self, self.image3)
        cv2.imshow('Output', output_img)
        cv2.waitKey()

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
                self.image_labels["image_3"].setPixmap(
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

    def in_paint_image(self):
        img1 = self.image1
        #image_path=self.file_path
        #original_image, mask = detect_object(self, image_path)
        #image = cv2.imread(image_path)
        model = YOLO("yolov8n.pt")
        # Run YOLO object detection
        results = model(self.image1)

        # Extract bounding boxes
        boxes = results[0].boxes.xyxy.numpy()  # Convert to numpy array

        # Create a mask to remove detected object
        mask = np.zeros(img1.shape[:2], dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill detected object

        inpainted_image = cv2.inpaint(img1, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        #output_img = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        # print('output image', output_img.shape)
        self.image3 = inpainted_image
        self.display_image(self, self.image3)
        #final_image = remove_object_and_fill(original_image, mask)

    def update_object_detection_video(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                results = model(frame)
                for result in results[0]:
                    box = result.boxes.xyxy.numpy()[0]  # Get bounding box coordinates
                    label = result.names[int(result.boxes.cls.numpy()[0])]  # Get object name

                    x1, y1, x2, y2 = map(int, box[:4])  # Convert coordinates to integers

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

                    # Put label text above the bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_labels["image_2"].setPixmap(
                    pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                                  Qt.KeepAspectRatio))

                cv2.imshow("TechVidvan", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    1
    def convert_to_grayscale(self):
        #if self.image1:
        original_image = self.image1
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # self.image3= gray_image
        #
        # self.display_image(self, gray_image)
        # Convert grayscale image to QPixmap for display

            # if self.image1:
        original_image = self.image1
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        h, w = gray_image.shape  # Ensure grayscale dimensions
        bytes_per_line = w  # Grayscale images use only one channel
        q_image = QImage(gray_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_labels["image_3"].setPixmap(pixmap)
        # self.image_labels["image_3"].setPixmap(
        #      pixmap.scaled(self.image_labels["image_3"].width(), self.image_labels["image_3"].height(),
        #                   Qt.KeepAspectRatio))

        # # Set the grayscale image in the image3 panel
        self.image_labels["image_3"].setPixmap(pixmap)
        # print("Grayscale conversion performed and displayed on Image 3.")


    def histogram_matching (self):
        reference=self.image2
        image=self.image1
        #reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        matched = match_histograms(image, reference, channel_axis=-1)
        self.image3 = matched
        self.display_image(self, self.image3)


    def mammogram(self):
        image=self.image1

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image is None:
            print("Error: Unable to load image. Check the file path.")
        else:
            # Ensure image is 8-bit unsigned integer
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(gray_image)

            # edges = cv2.Canny(enhanced_image, 50, 150)
            # # Convert edges to a color image (apply colormap)
            # color_mapped = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            h, w, ch = image1.shape
            bytes_per_line = ch * w
            q_image = QImage(image1.data, w, h, bytes_per_line, QImage.Format_RGB888)
            #
            # # Display directly in QLabel without saving
            pixmap = QPixmap.fromImage(q_image)
            self.image_labels["image_2"].setPixmap(
                pixmap.scaled(self.image_labels["image_2"].width(), self.image_labels["image_2"].height(),
                              Qt.KeepAspectRatio))

            # Apply thresholding to extract bright blobs
        _, bright_nodes = cv2.threshold(enhanced_image, 200, 255, cv2.THRESH_BINARY)

        # Label connected components (blobs)
        labeled_image = label(bright_nodes)

        # Configurable parameters
        # Configurable parameters
        eccentricity_threshold = 0.8  # Adjust: lower values favor circular shapes
        solidity_threshold = 0.2  # Adjust: closer to 1 favors solid shapes
        min_area = 2000  # Minimum blob area (adjustable)
        max_area = 20000  # Maximum blob area (adjustable)

        # Identify blobs that meet criteria
        selected_blobs = []
        for region in regionprops(labeled_image):
            print('area', region.area)
            if (region.eccentricity < eccentricity_threshold and
                    region.solidity > solidity_threshold and
                    min_area <= region.area <= max_area):
                print('area', region.area)
                selected_blobs.append(region)

        # Extract portion of image corresponding to selected blob and apply active contour
        if selected_blobs:
            blob = selected_blobs[0]  # Select the first suitable blob
            minr, minc, maxr, maxc = blob.bbox
            cropped_image = enhanced_image[minr:maxr, minc:maxc]

            # Create initial snake (active contour initialization)
            s = np.linspace(0, 2 * np.pi, 400)
            r = blob.centroid[0] - minr + 20 * np.sin(s)
            c = blob.centroid[1] - minc + 20 * np.cos(s)
            expansion_factor = 40  # Increase this value for a bigger initial contour

            s = np.linspace(0, 2 * np.pi, 400)
            r = blob.centroid[0] - minr + expansion_factor * np.sin(s)  # Higher radius
            c = blob.centroid[1] - minc + expansion_factor * np.cos(s)  # Higher radius

            init_snake = np.array([r, c]).T  # Transpose to match shape requirements

            # Apply Gaussian smoothing before snake segmentation
            smoothed_image = gaussian(cropped_image, 1)

            # Apply active contour model
            snake = active_contour(smoothed_image, init_snake, alpha=0.015, beta=10, gamma=0.0001)
            print(1)
            #contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            for point in snake:
                cv2.circle(image, (int(point[1] + minc), int(point[0] + minr)), 1, (0, 0, 255), -1)

            # Convert the processed image for PyQt display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.ascontiguousarray(image)

            #cv2.imshow('processed', image)
            #cv2.waitKey(0)
            #self.display_image(self,image)
            # self.image3=image
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            #
            # # Display directly in QLabel without saving
            pixmap = QPixmap.fromImage(q_image)
            self.image_labels["image_3"].setPixmap(
                pixmap.scaled(self.image_labels["image_3"].width(), self.image_labels["image_3"].height(),
                              Qt.KeepAspectRatio))


    def tyre_mark_detection(self):
        ROTATION_COUNT = 72
        # Load the image
        # Get image dimensions
        h, w = self.image1.shape[:2]
        center = (w // 2, h // 2)
        Im=self.image1
        for c in range(1, ROTATION_COUNT + 1):
            print(f"Processing {c}")  # Equivalent to `app.ProcessButton.Text = num2str(c)`
            # Compute rotation matrix
            try:
                Im = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
            except:
                1
            angle = -c * 5  # MATLAB's -c.*5 equivalent
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Rotate image (crop mode)
            IRot_original = cv2.warpAffine(Im, rotation_matrix, (w, h))
            IRot = IRot_original.copy()
            # Display rotated image (optional)
            # x_start, y_start, width, height = 900, 350, 550, 200
            x_start, y_start, width, height = 950, 380, 120, 180
            # Perform cropping
            im_roi = IRot[y_start:y_start + height, x_start:x_start + width]
            # im_roi = cv2.cvtColor(im_roi, cv2.COLOR_GRAY2RGB)
            # cv2.imshow('Final', im_roi)
            # cv2.waitKey(0)
            Ip = im_roi
            # Apply threshold

            Ip[Ip < 100] = 0
            # Apply dilation using a disk structuring element (similar to offsetstrel)
            se = disk(18)
            BW2 = cv2.dilate(Ip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
            # Compute Otsu threshold
            level = threshold_otsu(BW2)
            level = level + (0.1 * level)  # Adjust threshold like MATLAB
            # Binarize Image
            I_F = BW2 > level
            I_F = I_F.astype(np.uint8)  # Convert to uint8
            I_F = img_as_ubyte(I_F)  # Ensure correct format
            # I_F[I_F > 0] = 255
            I_F[(I_F >= 100) & (I_F <= 200)] = 0
            I_F[I_F > 210] = 0

            # Make binary (0 or 255)
            I_F = cv2.cvtColor(BW2, cv2.COLOR_BGR2GRAY) if BW2.ndim == 3 else BW2  # Convert to grayscale if needed
            # cv2.imshow('Final', I_F)
            # cv2.waitKey(0)
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((I_F), connectivity=8)
            # skimage_labels = label(labels)
            labels = label(labels)
            # Extract properties like MATLAB regionprops()
            regions = regionprops_table(labels, properties=['area', 'eccentricity', 'orientation', 'bbox', 'centroid',
                                                            'major_axis_length', 'minor_axis_length'])
            I_F_colored = cv2.cvtColor(I_F, cv2.COLOR_GRAY2BGR)
            # im_roi_coloured = cv2.cvtColor(im_roi, cv2.COLOR_GRAY2BGR)
            distances = [1]  # Pixel pair distance
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Different angles

            # Compute GLCM for each labeled region
            glcm_features = {}

            for i in range(len(regions['bbox-0'])):
                xmin = regions['bbox-1'][i]
                ymin = regions['bbox-0'][i]
                xmax = regions['bbox-3'][i]
                ymax = regions['bbox-2'][i]

                region_image = im_roi[ymin:ymax, xmin:xmax]
                if region_image.size > 0:
                    # Compute GLCM

                    glcm = graycomatrix(region_image.astype(np.uint8), distances, angles, symmetric=True, normed=True)

                    # Compute contrast and other texture features
                    contrast = graycoprops(glcm, 'contrast')
                    dissimilarity = graycoprops(glcm, 'dissimilarity')
                    homogeneity = graycoprops(glcm, 'homogeneity')
                    energy = graycoprops(glcm, 'energy')
                    correlation = graycoprops(glcm, 'correlation')

                    # Store features
                    glcm_features[i] = {
                        "contrast": contrast.mean(),
                        "dissimilarity": dissimilarity.mean(),
                        "homogeneity": homogeneity.mean(),
                        "energy": energy.mean(),
                        "correlation": correlation.mean(),
                    }
                    cv2.rectangle(im_roi, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Draw green boxes
            # for i in range(len(centroids)):
            #     cv2.circle(im_roi, (int(centroids[i][1]), int(centroids[i][0])), 3, (255, 0, 0),
            #                -1)  # Red dot for centroid
            #
            # cv2.imshow('Regions Highlighted', im_roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Convert to usable format
            # stats_df = pd.DataFrame(props)
            Stats = []
            for i in range(len(regions["area"])):
                stat = {
                    "Area": regions['area'][i],
                    "Orientation": regions['orientation'][i],
                    "eccentricity": regions['eccentricity'][i],
                    "Major": regions["major_axis_length"][i],
                    "Minor": regions["minor_axis_length"][i],
                    "BBX": regions["bbox-1"][i],
                    "BBY": regions["bbox-0"][i],
                    "CenX": regions["centroid-1"][i],
                    "CenY": regions["centroid-0"][i],
                    "BBXRange": regions["bbox-1"][i] + regions["bbox-3"][i],
                    "BBYRange": regions["bbox-0"][i] + regions["bbox-2"][i],
                }
                Stats.append(stat)
            # Display results
            # print(stats_df.head())  # Check region properties
            idx = [
                m for m, stat in enumerate(Stats)
                if (900 < stat["Area"] < 1000) and (-.9 < stat["Orientation"] < 0) and
                   (30 < stat["Major"] < 50) and (stat["Minor"] > 20) and (80 < stat['BBX'] < 90) and (
                               glcm_features[m]['contrast'] < 100)
            ]
            idx = [
                m for m, stat in enumerate(Stats)
                if (700 < stat["Area"] < 1000) and (-.9 < stat["Orientation"] < 1) and
                   (30 < stat["Major"] < 50) and (stat["Minor"] > 20)
                   and (glcm_features[m]['contrast'] < 100)
            ]


            IFinal_comp = img_as_ubyte(np.invert(BW2))  # Equivalent to `imcomplement` & `im2uint8`
            # Extract filtered stats
            stats_results = [Stats[z] for z in idx]
            if len(IRot_original.shape) == 2:  # If grayscale, convert to RGB
                IRot_original = cv2.cvtColor(IRot_original, cv2.COLOR_GRAY2BGR)
                1
            if len(idx) > 0:
                for k in range(len(stats_results)):
                    xbar = stats_results[k]["CenX"] + x_start
                    ybar = stats_results[k]["CenY"] + y_start
                    a = stats_results[k]["Major"] / 2
                    b = stats_results[k]["Minor"] / 2
                    theta = np.radians(stats_results[k]["Orientation"])

                    R = np.array([[np.cos(theta), np.sin(theta)],
                                  [-np.sin(theta), np.cos(theta)]])

                    phi = np.linspace(0, 2 * np.pi, 50)
                    cosphi = np.cos(phi)
                    sinphi = np.sin(phi)

                    xy = np.vstack((a * cosphi, b * sinphi))
                    xy = R @ xy

                    x = (xy[0, :] + xbar).astype(int)
                    y = (xy[1, :] + ybar).astype(int)

                    for i in range(len(x) - 1):
                        cv2.line(IRot_original, (x[i], y[i]), (x[i + 1], y[i + 1]), (0, 0, 255), 3)  # Draw ellipses on main image
                cv2.putText(IRot_original, "Tyre Mark Detected", (100, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                # Resized_IRot=cv2.resize(IRot_original, (900,900), interpolation=cv2.INTER_LINEAR
                self.image3=IRot_original
                self.display_image(self,IRot_original)
                cv2.imshow("Contours on Main Image", IRot_original)
                cv2.moveWindow("Contours on Main Image", 100, 100)
                cv2.namedWindow("Contours on Main Image", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Contours on Main Image", 400, 400)  # Set initial window size
                cv2.waitKey(0)
                cv2.destroyAllWindows(),
                break
            else:
                try:
                    Im = cv2.cvtColor(Im, cv2.COLOR_RGB2GRAY)
                except:
                    1
        cv2.putText(IRot_original, "Tyre Mark NOT Detected", (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Contours on Main Image", IRot_original)

    def Enhance_image(self):
        reference=self.image2
        image=self.image1
        #reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        matched = match_histograms(image, reference, channel_axis=-1)
        self.image3 = matched
        self.display_image(self, self.image3)


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

# Run the application
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet("background-color: lightblue;")
    main_window = ImageProcessingApp()
    main_window.show()
    sys.exit(app.exec_())
