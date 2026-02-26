'''
torch.cat: can concatenated multiple tensors along a specified dimension without changing the number of dimensions.
torch.stack: stacks multiple tensors along a new dimension, increasing the number of dimensions.Every dimension must have same shape

'''

import torch

torch.manual_seed(3)

t1 = torch.randint(1,10,(2,3))
print(f't1: {t1}, shape:{t1.shape}')

t2 = torch.randint(1,10,(2,3)) #can be (5,3), only the Concatenation dimension can be different
print(f't2: {t2}, shape: {t2.shape}')

#cat
t3 = torch.cat([t1,t2], dim=0) #(2,3) + (2,3) = (4,3)
print(f't3: {t3}, shape: {t3.shape}')

t4 = torch.cat([t1,t2], dim=1) #(2,3) + (2,3) = (2,6)
print(f't4: {t4}, shape: {t4.shape}')

#stack
t5 = torch.stack([t1,t2], dim=0)
print(f't5= {t5}, shape: {t5.shape}')

t6 = torch.stack([t1,t2], dim=1)
print(f't6= {t6}, shape: {t6.shape}')

t7 = torch.stack([t1,t2], dim=2)
print(f't7= {t7}, shape: {t7.shape}')