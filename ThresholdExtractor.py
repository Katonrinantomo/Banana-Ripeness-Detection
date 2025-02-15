import os
import cv2
import numpy as np

unripe_path = r'D:\College\A - Semester 4\IFB - 208 Pengolahan Citra Digital\TA PCD\train\unripe'
ripe_path = r'D:\College\A - Semester 4\IFB - 208 Pengolahan Citra Digital\TA PCD\train\ripe'
rotten_path = r'D:\College\A - Semester 4\IFB - 208 Pengolahan Citra Digital\TA PCD\train\rotten'

# Unripe (green)
lower_green = np.array([35, 40, 30])
upper_green = np.array([75, 255, 255])

# Ripe (yellow)
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([45, 255, 255])

# Rotten (dark)
lower_dark = np.array([0, 0, 0])
upper_dark = np.array([15, 255, 50])

def calculate_average_hsv(category_path, lower_bound, upper_bound):
    hsv_values = []
    for filename in os.listdir(category_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(category_path, filename)
            image = cv2.imread(img_path)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Apply mask to the HSV image
            masked_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

            # Calculate average HSV values of masked pixels
            avg_hsv = cv2.mean(hsv_image, mask=mask)[:3]  # Only take the first three values (H, S, V)
            hsv_values.append(avg_hsv)
    return np.mean(hsv_values, axis=0)

def calculate_thresholds(avg_hsv, margin):
    lower_threshold = avg_hsv - margin
    upper_threshold = avg_hsv + margin
    # Ensure that lower thresholds do not go below 0 and upper thresholds do not go above 255
    lower_threshold = np.clip(lower_threshold, 0, 255)
    upper_threshold = np.clip(upper_threshold, 0, 255)
    # Round the thresholds to the nearest integer
    lower_threshold = np.round(lower_threshold).astype(int)
    upper_threshold = np.round(upper_threshold).astype(int)
    return lower_threshold, upper_threshold

avg_hsv_unripe = calculate_average_hsv(unripe_path, lower_green, upper_green)
avg_hsv_ripe = calculate_average_hsv(ripe_path, lower_yellow, upper_yellow)
avg_hsv_rotten = calculate_average_hsv(rotten_path, lower_dark, upper_dark)

# Define initial margins for HSV thresholding (start with small margins)
hue_margin = 10
sat_margin = 40
val_margin = 40

# Calculate lower and upper thresholds for each category
lower_unripe, upper_unripe = calculate_thresholds(avg_hsv_unripe, np.array([hue_margin, sat_margin, val_margin]))
lower_ripe, upper_ripe = calculate_thresholds(avg_hsv_ripe, np.array([hue_margin, sat_margin, val_margin]))
lower_rotten, upper_rotten = calculate_thresholds(avg_hsv_rotten, np.array([hue_margin, sat_margin, val_margin]))

print(f"Unripe Thresholds: Lower: {lower_unripe}, Upper: {upper_unripe}")
print(f"Ripe Thresholds: Lower: {lower_ripe}, Upper: {upper_ripe}")
print(f"Rotten Thresholds: Lower: {lower_rotten}, Upper: {upper_rotten}")
