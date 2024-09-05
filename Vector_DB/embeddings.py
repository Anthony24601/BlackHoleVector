import matplotlib.pyplot as plt
import numpy as np

def read_image():
    file = open("images/3597_blur_avg.txt","r")
    lines = file.readlines()
    image = np.empty([180,180])
    t = []
    for line in lines:
        coord = line.strip().split()
        x = int(float(coord[0]) * 1000000)
        y = int(float(coord[1]) * 1000000)
        z = int(float(coord[2]) * 10000000000)
        image[x+90][y+90] = z
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    return image

def embed_image(image):
    pass



