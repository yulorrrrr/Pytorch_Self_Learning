'''
1. Row and Column Indexing
2. List Indexing 
3. Arrange Indexing 
4. Bool Indezing
5. multi-dim Indexing

'''

import torch

torch.manual_seed(24)

t1 = torch.randint(1, 10, (5,5))
print (f't1: {t1}')
print('-' * 30) 

#1. Row and Column Indexing [row, column]
print(t1[1])
print(t1[1, :])
print(t1[:, 2])
print('-' * 30)

#2. List Indexing 
print(t1[[0,1],[1,2]])
print(t1[[1,3],[2,4]]) #[1,2] & [3,4]
print(t1[[[0],[1]],[1,2]]) #four element: 0row 1,2 elements and 1row 1,2 elements
print('-' * 30)

#3. range Indexing 
print(t1[:3, :2]) #first 3 row first 2 column
print(t1[1:, :2]) #2-the last row, first 2 column
print(t1[1::2, 0::2]) #all odd row and even column

#4. Bool Indezing
print(t1[t1[:, 2]>5]) #Rows where the third column is greater than five
print(t1[:,t1[1]>5])  #Column where the fifth row is greater than five
print('-' * 10)

#5. multi-dim Indexing
t2 = torch.randint(1, 10, (2, 3, 4)) #two three row and four column matrix
print(f't2= {t2}')

#the first elementin dim0
print(f'{t2[0, :, :]}')

#in dim2
print(f'{t2[:, 0, :]}')

#in dim3
print(f'{t2[:, :, 0]}')