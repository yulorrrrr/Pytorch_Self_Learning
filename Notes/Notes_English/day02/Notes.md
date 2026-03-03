# Class Notes

## 1 Tensor Indexing Operations

```python
import torch

# Indexing starts from 0 when counting from left to right (0 -> first element)
# From right to left, it starts at -1
# data[row_index, column_index]
# data[axis0_index, axis1_index, axis2_index]

def dm01():
    # Create tensor
    torch.manual_seed(0)
    data = torch.randint(low=0, high=10, size=(4, 5))
    print('data->', data)

    # Get element by index
    # Row data: first row
    print('data[0] ->', data[0])

    # Column data: first column
    print('data[:, 0]->', data[:, 0])

    # Get values using index lists
    # Value at 2nd row 3rd column and 4th row 5th column
    print('data[[1, 3], [2, 4]]->', data[[1, 3], [2, 4]])

    # [[1], [3]]: values at (2nd row 3rd column, 2nd row 5th column)
    # and (4th row 3rd column, 4th row 5th column)
    print('data[[[1], [3]], [2, 4]]->', data[[[1], [3]], [2, 4]])

    # Boolean indexing
    # All row data where the second column is greater than 6
    print(data[:, 1] > 6)
    print('data[data[:, 1] > 6]->', data[data[:, 1] > 6])

    # All column data where the third row is greater than 6
    print('data[:, data[2]>6]->', data[:, data[2] > 6])

    # Range indexing (slicing) [start:end:step]
    # Rows 1 and 3, columns 2 and 4
    print('data[::2, 1::2]->', data[::2, 1::2])

    # Create 3D tensor
    data2 = torch.randint(0, 10, (3, 4, 5))
    print("data2->", data2)

    # First element along axis 0
    print(data2[0, :, :])

    # First element along axis 1
    print(data2[:, 0, :])

    # First element along axis 2
    print(data2[:, :, 0])


if __name__ == '__main__':
    dm01()
```

## 2 Tensor Shape Operations

### 2.1 Reshape

```python
import torch

# reshape(shape=(rows, columns)): Changes the shape of contiguous or non-contiguous tensors without changing data
# -1: automatically calculates dimension
# Example: (5, 6) -> (-1, 3)
# -1 * 3 = 5 * 6 -> -1 = 10 -> (10, 3)

def dm01():
    torch.manual_seed(0)
    t1 = torch.randint(0, 10, (5, 6))
    print('t1->', t1)
    print('t1 shape->', t1.shape)

    # Change shape to (2, 15)
    t2 = t1.reshape(shape=(2, 15))
    t3 = t1.reshape(shape=(2, -1))

    print('t2->', t2)
    print('t2 shape->', t2.shape)

    print('t3->', t3)
    print('t3 shape->', t3.shape)


if __name__ == '__main__':
    dm01()
```

### 2.2 squeeze & unsqueeze

```python
# squeeze(dim=):Removes dimensions of size 1
# If dim is specified, only removes that dimension if size=1
# If dim is not specified, removes all dimensions of size 1
# Example: (3,1,2,1) -> squeeze() -> (3,2)
# squeeze(dim=1) -> (3,2,1)

# unsqueeze(dim=):Adds a dimension of size 1 at specified position
# dim=-1 means last dimension

def dm02():
    torch.manual_seed(0)

    # 4D tensor
    t1 = torch.randint(0, 10, (3, 1, 2, 1))
    print('t1->', t1)
    print('t1 shape->', t1.shape)

    # squeeze: reduce dimension
    t2 = torch.squeeze(t1)
    print('t2->', t2)
    print('t2 shape->', t2.shape)

    # Specify dimension
    t3 = torch.squeeze(t1, dim=1)
    print('t3->', t3)
    print('t3 shape->', t3.shape)

    # unsqueeze: increase dimension
    # (3,2)->(1,3,2)
    # Last dimension (3,2)->(3,2,1)
    t4 = t2.unsqueeze(dim=-1)
    print('t4->', t4)
    print('t4 shape->', t4.shape)


if __name__ == '__main__':
    dm02()
```

### 2.3 transpose和permute

```python
# Swap dimensions

# torch.permute(input=, dims=): Change tensor dimension order arbitrarily
# dims: new dimension order (axis indices)
# Example: (1,2,3)->(3,1,2)

# torch.transpose(input=, dim0=, dim1=): Swap two dimensions
# Example: (1,2,3)->(2,1,3)
# Only swaps two dimensions at a time

def dm03():
    torch.manual_seed(0)
    t1 = torch.randint(low=0, high=10, size=(3, 4, 5))
    print('t1->', t1)
    print('t1 shape->', t1.shape)

    # Swap dimension 0 and 1
    t2 = t1.permute(dims=(1, 0, 2))
    print('t2->', t2)
    print('t2 shape->', t2.shape)

    # Change shape to (5, 3, 4)
    t3 = t1.permute(dims=(2, 0, 1))
    print('t3->', t3)
    print('t3 shape->', t3.shape)


if __name__ == '__main__':
    dm03()
```

### 2.4 view & contiguous

```python
# tensor.view(shape=): Changes shape of contiguous tensor (similar to reshape())

# tensor.is_contiguous(): Checks if tensor is contiguous (True/False)
# Tensor becomes non-contiguous after transpose/permute

# tensor.contiguous(): Converts tensor to contiguous memory layout

def dm04():
    torch.manual_seed(0)
    t1 = torch.randint(low=0, high=10, size=(3, 4))
    print('t1->', t1)
    print('t1 shape->', t1.shape)
    print('Is t1 contiguous->', t1.is_contiguous())

    # Change shape
    t2 = t1.view((4, 3))
    print('t2->', t2)
    print('t2 shape->', t2.shape)
    print('Is t2 contiguous->', t2.is_contiguous())

    # After transpose
    t3 = t1.transpose(dim0=1, dim1=0)
    print('t3->', t3)
    print('t3 shape->', t3.shape)
    print('Is t3 contiguous->', t3.is_contiguous())

    # Convert to contiguous before view
    t4 = t3.contiguous().view((3, 4))
    print('t4->', t4)

    t5 = t3.reshape(shape=(3, 4))
    print('t5->', t5)
    print('Is t5 contiguous->', t5.is_contiguous())


if __name__ == '__main__':
    dm04()
```

## 3 Tensor Concatenation Operations

### 3.1 cat/concat

```python
# torch.cat()/concat(tensors=, dim=): Concatenate along specified dimension
# Other dimensions must match
# Does not change total dimensions
# Specified dimension size increases

def dm01():
    torch.manual_seed(0)
    t1 = torch.randint(low=0, high=10, size=(2, 3))
    t2 = torch.randint(low=0, high=10, size=(2, 3))

    t3 = torch.cat(tensors=[t1, t2], dim=0)
    print('t3->', t3)
    print('t3 shape->', t3.shape)

    t4 = torch.concat(tensors=[t1, t2], dim=1)
    print('t4->', t4)
    print('t4 shape->', t4.shape)
```

### 3.2 stack

```python
# torch.stack(tensors=, dim=): Stack tensors along new dimension
# Adds new dimension (size equals number of tensors)
# Tensor dimensions change

def dm02():
    torch.manual_seed(0)
    t1 = torch.randint(low=0, high=10, size=(2, 3))
    t2 = torch.randint(low=0, high=10, size=(2, 3))

    t3 = torch.stack(tensors=[t1, t2], dim=0)
    print('t3->', t3)
    print('t3 shape->', t3.shape)

if __name__ == '__main__':
	dm02()
```

## 4 Autograd Module

### 4.1 梯度计算

```python
"""
Gradient: derivative, direction of fastest ascent/descent

Gradient Descent:W1 = W0 - lr * gradient

lr: learning rate (adjustable parameter)
W0: initial model weight

Important:
1. Loss must be scalar
2. Gradients accumulate by default
3. Stored in .grad attribute
"""

import torch

def dm01():
    # Create weight tensor
    # requires_grad: enable autograd
    # dtype must be floating point
    w = torch.tensor(data=[10, 20], requires_grad=True, dtype=torch.float32)

    loss = 2 * w ** 2
    print('loss->', loss)
    print('loss.sum()->', loss.sum())

    # Backpropagation (loss must be scalar)
    loss.sum().backward()

    print('w.grad->', w.grad)

    w.data = w.data - 0.01 * w.grad
    print('w->', w)
```

### 4.2 梯度下降法求最优解

```python
"""
Create an autograd-enabled weight tensor w
Define custom loss function: loss = w**2 + 20 (Later, no need to manually define - import loss functions for different problems)
Forward pass -> compute predicted y using previous model, then compute loss using loss function
Backward pass -> compute gradients
Gradient update -> update w using gradient descent
"""

import torch


def dm01():
    # Create autograd-enabled weight tensor w
    w = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
    print('w->', w)

    # Define custom loss function
    # Later, loss functions can be imported instead of manually defined
    loss = w ** 2 + 20
    print('loss->', loss)

    # 0.01 -> learning rate
    print('Start weight initial value: %.6f (0.01 * w.grad): None loss: %.6f' % (w, loss))

    for i in range(1, 1001):

        # Forward pass
        # Compute predicted value and then compute loss
        loss = w ** 2 + 20

        # Clear gradients
        # Gradients accumulate by default
        # If no gradient exists, it is None
        if w.grad is not None:
            w.grad.zero_()

        # Backward pass -> compute gradients
        loss.sum().backward()

        # Gradient update
        # W = W - lr * W.grad
        # w.data -> update tensor data in-place
        # Cannot directly assign to w (would create a new tensor)
        w.data = w.data - 0.01 * w.grad

        print('w.grad->', w.grad)
        print('Iteration:%d Weight w: %.6f, (0.01 * w.grad):%.6f loss:%.6f' % (i, w, 0.01 * w.grad, loss))

    print('w->', w, w.grad, 'minimum loss value', loss)


if __name__ == '__main__':
    dm01()
```

### 4.3 Important Notes on Gradient Computation

```python
# Autograd tensors cannot be directly converted to numpy arrays.
# Use detach() to create a new tensor that does not track gradients.

import torch


def dm01():
    x1 = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)
    print('x1->', x1)

    # Check whether tensor requires gradient
    # Returns True/False
    print(x1.requires_grad)

    # Use detach() to separate x1 from computation graph
    # Returns a new tensor
    # It does not require gradients
    # Shares data with original tensor
    x2 = x1.detach()

    print(x2.requires_grad)
    print(x1.data)
    print(x2.data)
    print(id(x1.data))
    print(id(x2.data))

    # Convert detached tensor to numpy array
    n1 = x2.numpy()
    print('n1->', n1)


if __name__ == '__main__':
    dm01()
```

### 4.4 Application of Autograd Module

```python
import torch
import torch.nn as nn  # # loss functions, optimizers, model modules


def dm01():
	def dm01():

    # todo: 1 - Define sample inputs x and targets y
    x = torch.ones(size=(2, 5))
    y = torch.zeros(size=(2, 3))
    print('x->', x)
    print('y->', y)

    # todo: 2 - Initialize model weights w and b (autograd tensors)
    w = torch.randn(size=(5, 3), requires_grad=True)
    b = torch.randn(size=(3,), requires_grad=True)
    print('w->', w)
    print('b->', b)

    # todo: 3 - Forward pass
    # Compute predicted y values
    y_pred = torch.matmul(x, w) + b
    print('y_pred->', y_pred)

    # todo: 4 - Compute loss using MSE loss function
    # Create MSE object (instantiate class)
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y)
    print('loss->', loss)

    # todo: 5 - Backward pass
    # Compute gradients for w and b
    loss.sum().backward()
    print('w.grad->', w.grad)
    print('b.grad->', b.grad)


if __name__ == '__main__':
	dm01()
```