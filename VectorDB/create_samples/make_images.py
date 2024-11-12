import numpy as np
from PIL import Image, ImageDraw
import os

# Set image dimensions and paths
img_size = (80, 80)
output_folder = './ellipses_test'
os.makedirs(output_folder, exist_ok=True)

# Define ellipse parameters
ellipse_params = [
    {'axes': (5, 40)},   # First ellipse type (small)
    {'axes': (10, 15)}    # Second ellipse type (new)
]

# Function to create an ellipse image with a specific intensity
def create_ellipse_image(img_size, ellipse_axes, intensity, rotation=0):
    img = Image.new('L', img_size, 0)  # Create grayscale image (0 = black)
    draw = ImageDraw.Draw(img)
    
    # Define ellipse bounding box (centered)
    box = [
        img_size[0] // 2 - ellipse_axes[0], img_size[1] // 2 - ellipse_axes[1],  # Top-left
        img_size[0] // 2 + ellipse_axes[0], img_size[1] // 2 + ellipse_axes[1]   # Bottom-right
    ]
    
    # Draw the ellipse
    draw.ellipse(box, fill=intensity)
    
    # Rotate the image by the specified angle
    img = img.rotate(rotation)
    
    return img

# Function to adjust intensity to ensure equal pixel sums
def adjust_intensity_to_match_total(img, target_sum):
    # Sum of current pixel values
    current_sum = np.sum(np.array(img))
    
    # If the current sum is 0 (completely black), return as is
    if current_sum == 0:
        return img
    
    # Scale the image to match the target pixel sum
    scaling_factor = target_sum / current_sum
    img = np.array(img).astype(np.float32) * scaling_factor
    img = np.clip(img, 0, 255)  # Ensure values are within grayscale range
    
    return Image.fromarray(img.astype(np.uint8))

# Main function to create images
def create_ellipse_images(num_images=100, target_pixel_sum=5000):
    for i, params in enumerate(ellipse_params):
        for img_idx in range(num_images):
            # Create random rotation between 0 and 360 degrees
            rotation = np.random.uniform(0, 360)
            
            # Create the ellipse with initial intensity 255 (white)
            img = create_ellipse_image(img_size, params['axes'], intensity=255, rotation=rotation)
            
            # Adjust the image intensity to match the target pixel sum
            img = adjust_intensity_to_match_total(img, target_pixel_sum)
            
            # Save the image
            img_filename = os.path.join(output_folder, f'ellipse_{i}_{img_idx}.png')
            img.save(img_filename)

# Generate images
create_ellipse_images(num_images=100, target_pixel_sum=100000)
