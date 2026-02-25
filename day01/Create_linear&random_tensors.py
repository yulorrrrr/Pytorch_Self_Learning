'''
torch.arange() and torch.linspace(): Create linear tensors

torch.random.initial_seed() and torch.random.manual_seed(): Set random seeds

torch.rand() / torch.randn(): Create random floating-point tensors

torch.randint(low, high, size=()): Create random integer tensors

'''

import torch

#1 create linear tensor
def dm01():
    # (1) Create a linear tensor with specified range
    t1 = torch.arange(0, 10, 2) #start end space
    print(f't1:{t1}, type:{type(t1)}')
    print('-' * 30)

    # (2) Create a linear tensor with specified range -> Arithmetic sequence
    t2 = torch.linspace(1, 10, 5) #start end number of element
    print(f't2:{t2}, type:{type(t2)}')

#2 create random tensor
def dm02():
    #step1: set random seed
    torch.manual_seed(3)

    #step2: create random tensor
    # (1) Random tensor following a uniform distribution
    t1 = torch.rand(size=(2,3))
    print(f't1:{t1}, type:{type(t1)}')
    print('-' * 30)

    # (2) Random tensor following a normal distribution
    t2 = torch.randn(size=(2,3))
    print(f't2:{t2}, type:{type(t2)}')
    print('-' * 30)

    # (3) create random int tensor
    t3 = torch.randint(low=1, high=10, size = (3,5))
    print(f't3:{t3}, type:{type(t3)}')

if __name__ == '__main__':
    #dm01()
    dm02()
    pass