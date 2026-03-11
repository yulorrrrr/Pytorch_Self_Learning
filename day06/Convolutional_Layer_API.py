"""
Example:
    Demonstrate the API of the convolution layer, used to extract local features of an image and obtain a feature map (Feature Map).

Introduction to Convolutional Neural Networks:
    Overview:
        Full name: Convolutional Neural Network (CNN), meaning: a neural network that contains convolutional layers.

Components:
    Convolutional Layer (Convolutional):
        Used to extract local features of an image.
        Implemented using convolution kernels (each convolution kernel = 1 neuron).
        The result after processing is called: Feature Map.
    
    Pooling Layer (Pooling): Used for dimensionality reduction and sampling.
    Fully Connected Layer (Full Connected, fc, linear): Used to predict results and output the final result.

Feature map size calculation:
    N = (W - F + 2P) / S + 1

    W: Size of the input image  
    F: Size of the convolution kernel  
    P: Size of padding  
    S: Stride  
    N: Size of the output image (feature map size)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#1. Define function, used for img loading and convolutional layer demonstration
def dm01():
    #1. load RGB image
    img = plt.imread('./day06/data/img.jpg')

    #2. print image and image shape
    print(f'img: {img}, img shape: {img.shape}') 

    #3. turn the image shape from HWC to CHW, img -> tensor -> change dimension
    img2 = torch.tensor(img, dtype=torch.float)
    img2 = img2.permute(2, 0, 1) #permute: change dimension order, from HWC to CHW
    #print(f'img2: {img2}, img2 shape: {img2.shape}')

    #4. save image. since we only have 1 imagem, we can add a dimension, from CHW to NCHW
    img3 = img2.unsqueeze(0) #unsqueeze: add a dimension, from CHW to NCHW
    #print(f'img3: {img3}, img3 shape: {img3.shape}')

    #5. create a convolutional layer object
    #param1: in_channels, param2: out_channels, param3: kernel_size, param4: stride, param5: padding
    conv = nn.Conv2d(3, 4, 3, 1, 0)

    #6. compute the convolution operation
    conv_img = conv(img3)

    #7. print the convolution result and its shape
    #print(f'conv_img: {conv_img}, conv_img shape: {conv_img.shape}')

    #8. check the 4 feature maps obtained after convolution
    img4 = conv_img[0]
    #print(f'img4: {img4}, img4 shape: {img4.shape}')

    #9. turn CHW to HWC, and plot the 4 feature maps
    img5 = img4.permute(1, 2, 0) #permute: change dimension order, from CHW to HWC
    #print(f'img5: {img5}, img5 shape: {img5.shape}')

    #10. plot the 4 feature maps
    feature1 = img5[:, :, 3].detach().numpy() #detach: separate from the computation graph, numpy: turn tensor to numpy array
    plt.imshow(feature1)
    plt.show()


#test
if __name__ == '__main__':
    dm01()
