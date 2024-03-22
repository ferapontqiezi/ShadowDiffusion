import os
import cv2
import numpy as np

# define folder path
image1_folder = './finaltest/'
image2_folder = './test/'
output_folder = './final_result/'

# define weight
weight_image1 = 0.5
weight_image2 = 0.5

# ensure folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Traverse image folders, load images and perform weighted averaging
for i in range(0, 75):  # final test: 75 images
    # load images
    image1_path = os.path.join(image1_folder, f'{i:04d}.png')
    image2_path = os.path.join(image2_folder, f'{i:04d}.png')
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Weighting images
    weighted_img = cv2.addWeighted(img1, weight_image1, img2, weight_image2, 0)

    # Save weighted average image
    output_path = os.path.join(output_folder, f'{i:04d}.png')
    cv2.imwrite(output_path, weighted_img)

print("Save weighted average image successfully!")
