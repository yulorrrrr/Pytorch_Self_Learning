import torch

#create specific type tensor
t1 = torch.tensor([1,2,3,4,5], dtype=torch.float) #defaut value: 32
print(f't1: {t1.dtype} element type: {type(t1)} tensor type: {type(t1)}')
print('-' * 30)