'''
API:
    reshape(): Change the shape of a tensor without changing its data.
    unsqueeze(): Add a dimension of size 1 at the specified position.
    squeeze(): Remove dimensions of size 1.
    transpose(): Swap two specified dimensions of a tensor.
    permute(): Reorder multiple dimensions of a tensor.
    view(): Return a new tensor with the same data but a different shape. Requires the tensor to be contiguous in memory.
    contiguous(): Return a contiguous tensor in memory. If the tensor is not contiguous (e.g., after transpose or permute), this function creates a copy with contiguous memory layout.
    is_contiguous(): Check whether a tensor is stored contiguously in memory.
'''

import torch

#1. reshape()
def dm01():
    t1 = torch.randint(1, 10, size=(2,3))
    print(f't1:{t1}, shape:{t1.shape}, row:{t1.shape[0]}, columns:{t1.shape[-1]}')

    #must have same number of elements
    t2 = t1.reshape(6,1)
    print(f't2:{t2}, shape:{t2.shape}, row:{t2.shape[0]}, columns:{t2.shape[-1]}') 
    #the order is the same as the origin

#2. unsqueeze(), squeeze()
def dm02():
    t1 = torch.randint(1, 10, size=(2,3))
    print(f't1:{t1}')

    #add a dim in 0 dimension
    t2 = t1.unsqueeze(0)
    print(f't2:{t2}, shapr: {t2.shape}')

    #add a dim in 1 dimension
    t3 = t1.unsqueeze(1)
    print(f't3:{t3}, shapr: {t3.shape}')

    #add a dim in 2 dimension
    t4 = t1.unsqueeze(2)
    print(f't4:{t4}, shapr: {t4.shape}')

    #delete all 1 dimension
    t5 = torch.randint(1, 10, size = (2, 1, 3, 1, 1))
    print(f't5: {t5}, shape: {t5.shape}')

    t6 = t5.squeeze()
    print(f't6: {t6}, shape: {t6.shape}')

#3. transpose() permute()
def dm03():
    t1 = torch.randint(1, 10, size=(2,3,4))
    print(f't1:{t1} shape: {t1.shape}')
    print('-' * 30)

    #(2,3,4) -> (4,3,2)
    t2 = t1.transpose(0, 2)
    print(f't2: {t2} shape: {t2.shape}')

    #(2,3,4) -> （4，2，3）
    t3 = t1.permute(2,0,1)
    print(f't3: {t3} shape: {t3.shape}')

#4. view(), contiguous(), is_contiguous()
def dm04():
    t1 = torch.randint(1, 10, size=(2,3))
    print(f't1: {t1}, shape: {t1.shape}')

    #identify if it is continuous
    print(t1.is_contiguous()) 

    #turn(2,3) -> (3,2)
    t2 = t1.view(3,2)
    print(f't1: {t2}, shape: {t2.shape}')

    #use transpose to change dim
    t3 = t1.transpose(0, 1)
    print(f't3: {t3}, shape:{t3.shape}')
    print(t3.is_contiguous()) 

    #turn t3 to continuous tensor through contiguous
    t4 = t3.contiguous().view(2,3)
    print(f't4: {t4}, shape:{t4.shape}')
    print(t4.is_contiguous())  

if __name__ == '__main__':
    dm04()