import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

img_size = (80, 80)
ellipse1_axes = (10, 20)
ellipse2_axes = (50, 20)
num_images = 100
save_path = './ellipses_test'

os.makedirs(save_path, exist_ok=True)

def create_ellipse_image(axes, rotation, img_size=img_size):
    img = np.zeros(img_size)

    # Define the center of the image
    center_x, center_y = img_size[1] // 2, img_size[0] // 2

    y, x = np.ogrid[:img_size[0], :img_size[1]]

    # Apply rotation matrix to the axes
    cos_angle = np.cos(rotation)
    sin_angle = np.sin(rotation)

    # Calculate the ellipse equation
    ellipse_eq = (((cos_angle * (x - center_x) + sin_angle * (y - center_y)) ** 2) / (axes[0] ** 2) +
                  ((sin_angle * (x - center_x) - cos_angle * (y - center_y)) ** 2) / (axes[1] ** 2))

    # Set pixels inside the ellipse to 1 (white)
    img[ellipse_eq <= 1] = 1

    return img

def save_image(img_array, filename):
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    img.save(filename)

def main():
    for i in range(num_images):
        # Generate random rotation angles
        rotation1 = np.random.uniform(0, 2 * np.pi)
        rotation2 = np.random.uniform(0, 2 * np.pi)

        # Create ellipses with random rotations
        img1 = create_ellipse_image(ellipse1_axes, rotation1)
        img2 = create_ellipse_image(ellipse2_axes, rotation2)

        # Save the images
        save_image(img1, os.path.join(save_path, f'ellipse0_{i}.png'))
        save_image(img2, os.path.join(save_path, f'ellipse1_{i}.png'))

main()
