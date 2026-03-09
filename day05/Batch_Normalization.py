'''
Example:
    Code demonstrates Batch Normalization.
    Batch normalization is also a type of regularization, and it is used to alleviate model overfitting.

Batch Normalization:
    Idea:
        First, standardize the data (remove some information), then scale the standardized data 
        (λ, interpreted as w, the weight parameter), and shift it (β, interpreted as b, the bias parameter),
        thereby restoring some information.

    Application Scenario:
        Batch normalization is widely used in computer vision tasks.

    BatchNorm1d:
        Mainly used in fully connected layers or networks that process one-dimensional data, such as text processing.
        It takes a tensor with shape (N, num_features) as input.

    BatchNorm2d:
        Mainly used in convolutional neural networks (CNNs) to process 2-D image data or feature maps.
        It takes a tensor with shape (N, C, H, W) as input.

    BatchNorm3d:
        Mainly used in 3-D convolutional neural networks (3D CNNs) to process three-dimensional data, such as videos or medical images.
        It takes a tensor with shape (N, C, D, H, W) as input.
'''
import torch
import torch.nn as nn
     
#1. Define function, deal with 2d data
def dm01():
    #1. create a tensor with shape (N, num_features)
    #1 picture, 2 channels, row = 3, column = 4
    input_2d = torch.randn(size=(1,2,3,4))
    print(f'input_tensor: {input_2d}')

    #2. create a BatchNorm2d object
    #param1: num_features, param2: eps(default: 1e-5), param3: momentum
    #param4: learnable parameters (affine) (λ，β)，scale and shift the normalized data   
    bn_2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True) 

    #3. apply batch normalization to the input tensor
    output_2d = bn_2d(input_2d)
    print(f'output_tensor: {output_2d}')

#2. Define function, deal with 1d data
def dm02():
    #1. create sample data
    #two row two column, each row is a sample, each column is a feature
    input_1d = torch.randn(size=(2, 2))
    print(f'input_tensor: {input_1d}')

    #2. create a BatchNorm1d object
    linear1 = nn.Linear(2, 4)

    #3. apply a linear transformation to the data
    l1 = linear1(input_1d)
    print(f'l1: {l1}')

    #4. apply batch normalization to the input tensor
    bn_1d = nn.BatchNorm1d(num_features=4)

    #5.apply batch normalization to the linear output l1
    output_1d = bn_1d(l1)
    print(f'output_tensor: {output_1d}')
    
#3. test
if __name__ == '__main__':
    #dm01()
    dm02()