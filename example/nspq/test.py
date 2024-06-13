import cv2
import numpy as np

# Load the black-and-white image
bw_image = cv2.imread('/usr/work/kafka/nspq/data_annots/ef9fe565-4bcc-40a1-ba77-2049d8585412.png', cv2.IMREAD_GRAYSCALE)

# Load the replacement image
rgb_image = cv2.imread('/usr/work/kafka/nspq/data_imgs/ef9fe565-4bcc-40a1-ba77-2049d8585412.png')

if bw_image.shape != rgb_image.shape[:2]:
    raise ValueError("The black-and-white image and the RGB image must have the same dimensions")

# Create a mask from the black-and-white image where white is considered 255
mask = bw_image == 255

# Create a new image to hold the result
result_image = rgb_image.copy()

# Overwrite the white parts in the black-and-white image onto the RGB image
result_image[mask] = (255, 255, 255)  # Overwrite with white color, or choose another color if needed

# Save or display the result
cv2.imwrite('result_image.jpg', result_image)

