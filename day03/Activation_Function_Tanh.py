'''
Plot the function curve and derivative curve of the Tanh activation function.


Introduction to activation functions:

    Purpose of activation functions:
        Add nonlinearity to the model, enabling neural networks to perform classification as well as solve regression problems.

    Common activation functions:
        Sigmoid
        ReLU
        Tanh
        Softmax


    Sigmoid activation function:
        1. Mainly used in the output layer for binary classification tasks. Also suitable for shallow neural networks (no more than 5 layers).
        2. For input values in the range [-6, 6], it performs well. In the range [-3, 3], the effect is especially significant.
        3. For larger input values, the gradient becomes very small (vanishing gradient problem).
        4. It maps the data values to the range [0, 1].

    Tanh activation function:
        1. Mainly used in the hidden layer. Also suitable for shallow neural networks (no more than 5 layers).
        2. For input values in the range [-3, 3], it performs well. In the range [-1, 1], the effect is especially significant.
        3. For larger input values, the gradient becomes very small (vanishing gradient problem).
        4. It maps the data values to the range [-1, 1].


Note:
    If the following error occurs when plotting the activation function graph:
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. You need to delete the libiomp5md.dll file located in:anaconda3/Lib/site-packages/torch/lib
'''
import torch
import matplotlib.pyplot as plt

#1. Create figure and axis, 1row 2column
fig, axes = plt.subplots(1,2)

#2. Generate 1000 data points between -20 and 20
x = torch.linspace(-20, 20, 1000)
#print(f'x: {x}')

#3. Compute Tanh value for these points
y = torch.tanh(x)
print(f'y: {y}')

#4. Plot the Tanh function on the first subplot
axes[0].plot(x, y)
axes[0].set_title('Tanh Function')
axes[0].grid()

#5. Plot the derivative of the Tanh function on the second subplot

#5.1 Regenerate 1000 data points between -20 and 20
#param1: start value / param2: end value / param3: number of elements / param4: whether gradient is required
x = torch.linspace(-20, 20, 1000, requires_grad=True)

#5.2 Specifically compute the derivative values of the Tanh activation function for the above 1000 points.
torch.tanh(x).sum().backward()

#5.3 draw the figure
axes[1].plot(x.detach(), x.grad)
axes[1].set_title('Derivative plot of the Tanh activation function')
axes[1].grid()
plt.show()