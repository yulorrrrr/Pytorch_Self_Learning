'''
Image Types:
Binary image:      1 channel, each pixel consists of 0 or 1.
Grayscale image:   1 channel, pixel values range from [0, 255].
Indexed image:     1 channel, pixel values range from [0, 255], where each pixel represents an index in the color table.
RGB color image:   3 channels: Red, Green, and Blue.
'''
import numpy as np
import matplotlib.pyplot as plt
import torch

#1. Define function, plot completely back and completely white image
def dm01():
    #1. define completely black image, 0: completely black, 255: completely white
    #HWC: height, width, channel
    img1 = np.zeros((200, 200, 3)) #
    
    #Plot
    plt.imshow(img1)
    plt.axis('off') #turn off axis
    plt.show()

    #2. define completely white image, 0: completely black, 255: completely white
    img2 = np.ones((200, 200, 3)) 

    #Plot
    plt.imshow(img2)
    plt.show()

#2. Define function, load image
def dm02():
    #1. load image
    img = plt.imread('./day06/data/img.jpg')
    print(f'img: {img}, img shape: {img.shape}')

    #2. save image
    plt.imsave('./day06/data/img_copy.jpg', img)
    

#3. Test
if __name__ == '__main__':
    #dm01()
    dm02()