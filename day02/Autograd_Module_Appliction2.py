'''
Example:
Demonstrate the automatic differentiation module and explain how to compute gradients and update parameters.

Task:
Find the minimum value of the function 
y= = x**2 + 20, and print the corresponding value of y when w at the minimum (i.e., the optimized parameter).

Solution Steps:

1.Define a variable: w = 10, requires_grad=True, dtype=torch.float32
2.Define the function: loss = w ** 2 + 20
3. Use gradient descent to iterate 1000 times to find the optimal solution.
    3.1 Perform forward computation.
    3.2 Clear the gradient: w.grad.zero_()
    3.3 Perform backpropagation.
    3.4 Update the parameter:
    w.data = w.data - 0.01 * w.grad
'''

import torch

#1.Define a variable: w = 10, requires_grad=True, dtype=torch.float32
w = torch.tensor(10, requires_grad=True, dtype=torch.float32)

#2.Define the function: loss = w ** 2 + 20
loss = w ** 2 + 20 # loss' = 2w

print(f'the initial value of weight: {w}, loss: {loss} ')

#iterate 1000 times
for i in range(1, 101):
    
    #3.1 Perform forward computation.
    loss = w ** 2 + 20

    #3.2 Clear the gradient: w.grad.zero_()
    if w.grad is not None:
        w.grad.zero_()

    #3.3 Perform backpropagation.
    loss.sum().backward()

    #3.4 Update the parameter:
    #print(f'grad: {w.grad}')
    w.data = w.data - 0.01 * w.grad

    #3.5 print
    print(f'No.{i}, weight:{w}, (0.01 * w.grad):{0.01*w.grad:.5f}, loss:{loss:.5f}')

#4. print final result
print(f'weight:{w}, grad:{w.grad:.5f}, loss: {loss:.5f}')
