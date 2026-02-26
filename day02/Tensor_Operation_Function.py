'''
Badic Operation functon:
API:
    sum(), max(), min(), mean() -> have dim, 1:colunm 0: row
    pow(), sqrt(), exp(), log(), log2(), log10() -> don't have dim
'''

import torch

t1 = torch.tensor([
    [1,2,3],
    [4,5,6]
], dtype=torch.float)
#sum()
print(f't1:{t1}')
print(t1.sum(dim=0))
print(t1.sum(dim=1))
print(t1.sum())
print('-' * 30)

#max()
print(t1.max(dim=0))
print(t1.max(dim=1))
print(t1.max())
print('-' * 30)

#mean
print(t1.mean(dim=0))
print(t1.mean(dim=1))
print(t1.mean())
print('*' * 30)

#no dim function
print(t1 ** 2)
print(t1.sqrt())
print(t1.exp())
print(t1.log())