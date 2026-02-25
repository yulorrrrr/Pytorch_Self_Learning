"""

Tensor:

The PyTorch framework is one of the most commonly used deep learning frameworks.
Whether it is ANN (Artificial Neural Networks), CNN (Convolutional Neural Networks),
or RNN (Recurrent Neural Networks) , tensors are used at the underlying level to process data.

Tensor → A container that stores elements of the same type, and the elements must be numerical values.

Basic ways to create tensors:

    torch.tensor      Create a tensor based on specified data
    torch.Tensor      Create a tensor based on shape; it can also be used to create a tensor with specified data
    torch.IntTensor torch.FloatTensor  torch.DoubleTensor  Create tensors with specific data types

*tips: Unlike tensor, Tensor can create a tensor directly by specifying its shape.
"""

import torch
import numpy as np
#1 torch.tensor 

def dm01():
    # (1) scalar tensor
    t1 = torch.tensor(10)
    print(f't1:{t1} type:{type(t1)}')
    print('-' * 30)

    # (2) list -> tensor
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f't2:{t2} type:{type(t2)}')
    print('-' * 30)

    # (3) numpy nd -> tensor
    data = np.random.randint(0, 10, size =  (2,3))
    t3 = torch.tensor(data, dataType=torch.float)
    print(f't3:{t3}, type:{type(t3)}]')

#2 torch.Tensor

def dm02():
    # (1) scalar tensor
    t1 = torch.Tensor(10)
    print(f't1:{t1} type:{type(t1)}')
    print('-' * 30)

    # (2) list -> tensor
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f't2:{t2} type:{type(t2)}')
    print('-' * 30)

    # (3) numpy nd -> tensor
    data = np.random.randint(0, 10, size =  (2,3))
    t3 = torch.Tensor(data)
    print(f't3:{t3} type:{type(t3)}')
    print('-' * 30)

    #create directly
    t4 = torch.Tensor(2,3)
    print(f't4:{t4} type:{type(t4)}')
    print('-' * 30)

#3 torch.IntTensor torch.FloatTensor  torch.DoubleTensor

def dm03():
    # (1) scalar tensor
    t1 = torch.IntTensor(10)
    print(f't1:{t1} type:{type(t1)}')
    print('-' * 30)

    # (2) list -> tensor
    data = [[1,2,3],[4,5,6]]
    t2 = torch.IntTensor(data)
    print(f't2:{t2} type:{type(t2)}')
    print('-' * 30)

    # (3) numpy nd -> tensor
    data = np.random.randint(0, 10, size =  (2,3))
    t3 = torch.IntTensor(data)
    print(f't3:{t3} type:{type(t3)}')
    print('-' * 30)

    #create directly
    t4 = torch.IntTensor(2,3)
    print(f't4:{t4} type:{type(t4)}')
    print('-' * 30)

#4 test
if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()
    pass