'''
Background:

Weight update formula:
W_new = W_old - learning rate * gradient

The gradient is the derivative of the loss function.

Regarding the derivative of the loss function, we usually do not compute it manually. 
Since it is very common in practice, PyTorch provides an automatic differentiation module 
internally, which is specifically designed to compute gradients for different loss functions.

This allows us to perform backpropagation to update the weight parameters w and bias parameters 

**Only scalar tensors can be differentiated. Most of the operation use float 
'''

import torch

#1.create tensor
w = torch.tensor(10, requires_grad=True, dtype=torch.float)

#2.define loss to represent loss value
loss = 2* w **2 #loss = 2w^2 -> differentiate: 4w

#3.print type
#print(f'type: {type(loss.grad_fn)}')  #<class 'MulBackward0'>
#print(loss.sum())

#4.Differentiate
#loss.backward()
loss.sum().backward() #make sure loss is scaler

#5.substitute into weight update formula
w.data = w.data-0.01 * w.grad

#print final result
print(f'new weight: {w}')