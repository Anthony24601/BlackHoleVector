import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create directory to save images
output_dir = "clustering_images"
os.makedirs(output_dir, exist_ok=True)

# Function to generate points on a circle
def generate_circle_points(num_points, radius=1, offset=(0, 0, 0)):
    a = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(a) + offset[0]
    y = radius * np.sin(a) + offset[1]
    z = offset[2] * np.ones(num_points)
    return np.vstack((x, y, z)).T

# Updated function to map 3D points to 80x80 grayscale images with normalized intensity
def map_point_to_image_with_grayscale(point, image_size=80, circle_radius=3, total_intensity=1000):
    img = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Simple projection to 2D
    x_2d = int(((point[0] + 2) / 4) * image_size)
    y_2d = int(((point[1] + 2) / 4) * image_size)
    
    x_2d = np.clip(x_2d, 0, image_size - 1)
    y_2d = np.clip(y_2d, 0, image_size - 1)
    
    # Draw a small circle around the projected point
    for dx in range(-circle_radius, circle_radius + 1):
        for dy in range(-circle_radius, circle_radius + 1):
            if dx**2 + dy**2 <= circle_radius**2:
                new_x = np.clip(x_2d + dx, 0, image_size - 1)
                new_y = np.clip(y_2d + dy, 0, image_size - 1)
                img[new_y, new_x] = 1.0  # Set pixel intensity to 1
    
    # Normalize the image so that the total intensity is equal to `total_intensity`
    img_sum = np.sum(img)
    if img_sum > 0:
        img = img * (total_intensity / img_sum)
    
    # Convert to uint8 for saving as image
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img

# Function to generate and save grayscale normalized images for both clusters
def generate_grayscale_normalized_images():
    num_points = 100
    total_intensity = 1000  # Set desired total pixel intensity for all images
    
    # First cluster (circle 1)
    cluster1_points = generate_circle_points(num_points, radius=1, offset=(0, 0, 0))
    
    # Second cluster (circle 2)
    cluster2_points = generate_circle_points(num_points, radius=1, offset=(1, 0, 0))
    
    # Save images
    for i, point in enumerate(cluster1_points):
        img = map_point_to_image_with_grayscale(point, total_intensity=total_intensity)
        img_pil = Image.fromarray(img)
        img_pil.save(f"{output_dir}/cluster1_grayscale_image_{i+1}.png")
    
    for i, point in enumerate(cluster2_points):
        img = map_point_to_image_with_grayscale(point, total_intensity=total_intensity)
        img_pil = Image.fromarray(img)
        img_pil.save(f"{output_dir}/cluster2_grayscale_image_{i+1}.png")

# Generate and save the images
generate_grayscale_normalized_images()

print(f"Images saved to directory: {output_dir}")

"""
1. Create two clusters in R3
    First cluster (ring 1): Parametric equation (cos⁡a,sin⁡a,0)(cosa,sina,0), where 0≤a<2π0≤a<2π.
    Second cluster (ring 2): Parametric equation (1+cos⁡b,0,sin⁡b)(1+cosb,0,sinb), where 0≤b<2π0≤b<2π.



"""
