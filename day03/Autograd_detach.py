'''
Autograd = differentiate, which means to calculate grad base on loss
weight update formula: w_new = w_old - learning_rate * gradient

Once a tensor has automatic differentiation enabled, it cannot be directly converted 
into a NumPy ndarray; it must first be detached using the detach() function.
'''
import torch
import numpy as np 

t1=torch.tensor([10,20],requires_grad=True, dtype=torch.float)
print(f't1: {t1}, type: {t1.type}')

#try to convert tensor to numpy
#n1 = t1.numpy() #error

#method: use detach()
t2 = t1.detach()
print(f't2: {t2}, type: {t2.type}')

#test if t1 and t2 share same memory
t1.data[0]=100
print(f't1: {t1}, type: {t1.type}')
print(f't2: {t2}, type: {t2.type}')
print('-' * 30)

#Check whether t1 and t2 support automatic differentiation.
print(f't1: {t1.requires_grad}, t2: {t2.requires_grad}') #t1 true, t2 false
print('-' * 30)

#convert t2 to numpy
n1 = t2.numpy()
print(f'n1: {n1}, type: {type(n1)}')

#simple way
n2 = t1.detach().numpy
print(f'n2: {n2}, type: {type(n2)}')