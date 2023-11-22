import cv2
import numpy as np

# Load the image
img = cv2.imread('cv.png')

# Convert image to float32
img_float32 = np.float32(img)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners using Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate corners to mark them
dst = cv2.dilate(dst, None)

# Mark the corners on the original image
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Display the corners
cv2.imshow('Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
