'''
#1 tensor -> numpy ndarray

scalar_tensor.numpy(): Share memory
scalar_tensor.numpy().copy(): Do not share memory; chain-style programming

#2 NumPy ndarray → Tensor

torch.from_numpy() -> Share memory
torch.tensor(ndarray) -> Do not share memory

#3 Extract the value from a scalar tensor

scalar_tensor.item()

'''
import torch
import numpy as np

#1. define function: tensor-> numpy
def dm01():
    t1 = torch.tensor([1,2,3,4,5])
    #n1 = t1.numpy()  #share memory
    n1 = t1.numpy().copy()  #don't share memory
    print(f'n1: {n1}, type: {type(n1)}')
    n1[0] = 100
    print(f'n1: {n1}')
    print(f't1: {t1}')

#2. numpy-> tensor
def dm02():
    n1 = np.array([11,22,33])
    print(f'n1: {n1}, type: {type(n1)}')

    t1 = torch.from_numpy(n1) #share memory
    print(f't1: {t1}, type: {type(t1)}')

    t2 = torch.tensor(n1) #don't share memory
    print(f't2: {t2}, type: {type(t2)}')

    n1[0]=100
    print(f'n1: {n1}')
    print(f't1: {t1}')
    print(f't2: {t2}')

#3.Extract the value from a scalar tensor
def dm03():
    t1 = torch.tensor(100) #can only be bool or number
    #t1 = torch.tensor([100,200]) #can only have one elemenr=t
    print(f't1: {t1}, type:{type(t1)}')

    a = t1.item()
    print(f'a: {a}, type:{type(a)}')

if __name__ == '__main__':
    #dm01()  
    #dm02()  
    dm03()   