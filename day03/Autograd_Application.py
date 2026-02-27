'''
real-world use cases of auto-grad.

Conclusion:
1. Forward pass to compute predicted values(z)
2. Based on the loss function, combine the predicted value(z) with the true value(y) to compue the loss
3. Base on the weight update formula: w_new = w_old - learning_rate * gradient to compute weight

'''

import torch

#define x, Represents the features (input data). Hypothesis: 2 rows 5 columns, all ones matrix
x = torch.ones(2,5)
print(f'x: {x}')

#define y, Represents the labels (ground truth)., Hypothesis: 2 rows 3 columns, all zeros matrix
y = torch.zeros(2,3)
print(f'y: {y}')

#Initialize weight
w = torch.randn(5,3,requires_grad=True) 
print(f'w: {w}')

b = torch.randn(3, requires_grad=True)
print(f'b: {b}')

#compute y through forward pass
z = torch.matmul(x,w) + b
#z = x @ w + b
print(f'z: {z}')

#define loss
criterion = torch.nn.MSELoss() #neural network
loss = criterion(z,y)
print(f'loss: {loss}')

#Compute gradients via autograd and update weights through backpropagation.
loss.backward()

#print new w,b
print(f'w grad: {w.grad}')
print(f'b grad: {b.grad}')

#w_new = w_old - learning_rate * gradient
