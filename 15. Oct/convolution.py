import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image

image = Image.open('15. Oct/butterfly.png')
image = np.array(image)

filter_1 = np.array([[-1, -1, -1], 
                     [-1,  4, -1], 
                     [-1, -1, -1]])

filter_2 = np.array([[-1,  0, -1], 
                     [ 0,  4,  0], 
                     [-1,  0, -1]])

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

red_conv_1 = convolve(red_channel, filter_1, mode='reflect')
green_conv_1 = convolve(green_channel, filter_1, mode='reflect')
blue_conv_1 = convolve(blue_channel, filter_1, mode='reflect')

red_conv_2 = convolve(red_channel, filter_2, mode='reflect')
green_conv_2 = convolve(green_channel, filter_2, mode='reflect')
blue_conv_2 = convolve(blue_channel, filter_2, mode='reflect')

conv_result_1 = np.stack((red_conv_1, green_conv_1, blue_conv_1), axis=-1)
conv_result_2 = np.stack((red_conv_2, green_conv_2, blue_conv_2), axis=-1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(np.clip(conv_result_1, 0, 255).astype(np.uint8))
plt.title("Filter 1")

plt.subplot(1, 3, 3)
plt.imshow(np.clip(conv_result_2, 0, 255).astype(np.uint8))
plt.title("Filter 2")

plt.show()
