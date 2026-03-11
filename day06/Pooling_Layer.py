"""
Example:
    Demonstrate operations related to the pooling layer.

Pooling Layer (Pooling):
    Purpose:
        Dimensionality reduction.

    Approach:
        Max pooling.
        Average pooling.

    Characteristic:
        Pooling does not change the number of channels of the data.
"""

import torch
import torch.nn as nn

#1. Define function, demonstrate single channel pooling
def dm01():
    #1. create a one channel, 3 * 3 two dimensional tensor
    inputs = torch.tensor([[
        [0,1,2],
        [3,4,5],
        [6,7,8]
    ]])

    #print(f'inputs: {inputs}, inputs shape: {inputs.shape}') #(1,3,3)

    #2. create a max pooling layer object
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output1 = pool1(inputs)
    print(f'max pooling output: {output1}, max pooling output shape: {output1.shape}') # (1,2,2)

    #3. create an average pooling layer object
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output2 = pool2(inputs)
    print(f'average pooling output: {output2}, average pooling output shape: {output2.shape}') # (1,2,2)

#2. Define function, demonstrate multi-channel pooling
def dm02():
    #1. create a two channel, 3 * 3 two dimensional tensor
    inputs = torch.tensor([
        [
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ],
        [
            [10,20,30],
            [40,50,60],
            [70,80,90]
        ],
        [
            [11,22,33],
            [44,55,66],
            [77,88,99]
        ]
    ])

    #print(f'inputs: {inputs}, inputs shape: {inputs.shape}') #(2,3,3)

    #2. create a max pooling layer object
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output1 = pool1(inputs)
    print(f'max pooling output: {output1}, max pooling output shape: {output1.shape}') # (2,2,2)

    #3. create an average pooling layer object
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output2 = pool2(inputs)
    print(f'average pooling output: {output2}, average pooling output shape: {output2.shape}') # (2,2,2)

#test
if __name__ == '__main__':
    #dm01()
    dm02()