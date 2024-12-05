

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("dental.jpg")  # Replace "example.jpg" with your image path
if image is None:
    raise ValueError("Image not found or path is incorrect!")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection (Canny)
edges = cv2.Canny(gray_image, threshold1=100, threshold2=250)

# Find contours from edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to visualize contours
contour_image = np.zeros_like(image)

# Filter contours based on area
min_area = 100  # Set the minimum area threshold
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


# Draw contours on the blank image
for idx, contour in enumerate(contours):
    # Random color for each region
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    cv2.drawContours(contour_image, [contour], -1, color, thickness=cv2.FILLED)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Edges")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmented Regions")
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
