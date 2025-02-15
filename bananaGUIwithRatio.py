import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore

import cv2
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('Classification_GUI.ui', self)
        self.insert_button = self.findChild(QtWidgets.QPushButton, 'insert_button')
        self.start_button = self.findChild(QtWidgets.QPushButton, 'start_button')
        self.image1 = self.findChild(QtWidgets.QLabel, 'image1')
        self.insert_button.clicked.connect(self.insert_image)
        self.start_button.clicked.connect(self.start_classification)
        self.image_path = None
    

    def insert_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        self.image_path = file_name
        if file_name:
            pixmap = QPixmap(file_name)
            self.image1.setPixmap(pixmap.scaled(self.image1.size(), QtCore.Qt.KeepAspectRatio))


    def start_classification(self):
        image = cv2.imread(self.image_path)

        resized_image = self.resize_image(image)
        banana_category, contours, rotten_percentage = self.extract_banana(resized_image)

        if banana_category == "Unripe":
            category_text = "Unripe banana detected."
            color = (0, 255, 0)  # Green color for unripe
        elif banana_category == "Ripe":
            category_text = "Ripe banana detected."
            color = (0, 255, 255)  # Yellow color for ripe
        elif banana_category == "Rotten":
            category_text = f"Rotten banana detected. {rotten_percentage:.2f}% rotten."
            color = (0, 0, 255)  # Red color for rotten
        else:
            category_text = "No banana detected in the image."
            color = (255, 255, 255)  # White color for no banana

        # Draw bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), color, 2)

        # Add text indicating detected category
        cv2.putText(resized_image, category_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Detected Bananas', resized_image)

    def resize_image(self, image, max_width=500):
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / width
            resized_image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
            return resized_image
        else:
            return image

    def extract_banana(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Unripe (green)
        lower_green = np.array([28, 99, 98])
        upper_green = np.array([48, 179, 178])

        # Ripe (yellow)
        lower_yellow = np.array([15, 139, 137])
        upper_yellow = np.array([35, 219, 217])

        # Rotten (dark)
        lower_dark = np.array([0, 36, 0])
        upper_dark = np.array([16, 116, 66])

        # Combine masks
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_dark, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Determine the category based on the contour areas
        if contours_green:
            area_green = max(cv2.contourArea(contour) for contour in contours_green)
        else:
            area_green = 0

        if contours_yellow:
            area_yellow = max(cv2.contourArea(contour) for contour in contours_yellow)
        else:
            area_yellow = 0

        if contours_dark:
            area_dark = max(cv2.contourArea(contour) for contour in contours_dark)
        else:
            area_dark = 0

        # Calculate total area
        total_area = area_green + area_yellow + area_dark

        # Calculate the ratio of dark area to total area
        if total_area > 0:
            ratio_dark = area_dark / total_area
            rotten_percentage = ratio_dark * 100
        else:
            rotten_percentage = 0

        # Return the category based on area and ratio
        if ratio_dark > 0.5:  # Adjust the threshold as needed
            return "Rotten", contours_dark, rotten_percentage
        elif area_yellow > area_green and area_yellow > area_dark:
            return "Ripe", contours_yellow, rotten_percentage
        elif area_green > area_yellow and area_green > area_dark:
            return "Unripe", contours_green, rotten_percentage
        else:
            return "No Banana", [], rotten_percentage
                    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())