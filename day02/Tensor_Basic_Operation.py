'''
API:
    add(), sub(),mul(),div(),neg() -> +-*/ turn to negative
    add_(), sub_(),mul_(),div_(),neg_() -> Can modify  original data

Dot product: The two tensors must have the same shape
    API:
        t1*t2
        t1.mul(t2)

Matric product: The column of A is the row or B
    API:
        t1 @ t2
        t1.matmul(t2)
'''

import torch

t1 = torch.tensor([1,2,3])

#t2 = t1.add(10)
#t2 = t1 +10 #don't modift original data

#t1.add_(10)
#t1 += 10 #will modify original data

t2 = t1.add(10)
t2 = t1.neg()

print(f't1: {t1}')
print(f't2: {t2}')

def dm01(): #dot
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't1:{t1}')

    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't2: {t2}')

    t3 = t1 * t2
    print(f't3: {t3}')

def dm02(): #matrix
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't1:{t1}')

    t2 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(f't2: {t2}')

    t3 = t1 @ t2
    print(f't3: {t3}')

if __name__ == '__main__':
    dm02()