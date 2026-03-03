# Class Notes

## 1 Introduction to Deep Learning

### 1.1 Concept of Deep Learning

- Deep learning is a category of machine learning algorithms that use artificial neural networks as their structure and can automatically extract features.
- The core idea of deep learning is to use artificial neural networks to automatically extract features.

### 1.2 Characteristics of Deep Learning

- Automatic feature extraction  
- Poor interpretability  
- Requires large amounts of data and high-performance computing power  
- Nonlinear transformations (introducing nonlinear factors)

### 1.3 Deep Learning Models

- ANN (Artificial Neural Network) – Perceptron  
- CNN (Convolutional Neural Network) – Image/Video  
- RNN (Recurrent Neural Network) – NLP  
- Transformer – Derived from RNN  
- Custom-built learners 
- ...

### 1.4 Application Scenarios of Deep Learning

- Natural Language Processing (NLP)
  - Generative AI (AIGC) / Large Language Models
  - Machine Translation
  - Speech Recognition
  - ...

- Computer Vision (CV)
  - Image Recognition
  - Face Unlock
  - Video Synthesis
  - ...

- Recommendation Systems
  - Movies
  - Music
  - Articles
  - Videos
  - Products

---

## 2. Introduction to the PyTorch Framework

- PyTorch is a deep learning framework and a third-party Python package. Data is stored in tensor format.

- Features of PyTorch:
  - Data type is tensor
  - Automatic differentiation module (automatic gradient computation)
  - Can run on GPU / TPU / NPU for acceleration
  - Compatible with various platforms, systems, and hardware (GPUs)
  - Currently updated to version 2.5

---

## 3. Tensor Creation

### 3.1 What is a Tensor

- A tensor is essentially a matrix and can be multi-dimensional:
  - 0D → Scalar  
  - 1D → [1 2 3 4 5]  
  - 2D → [[1 2 3], [4 5 6]]  
  - 3D → ...

- A tensor is an object created through a class and provides various methods and attributes.

### 3.2 Basic Creation Methods

- torch.tensor(data=): specific data
- torch.Tensor(data=, size=): specific data or shape
- torch.IntTensor(data=) / FloatTensor(): specific data

  ```python
  import torch
  import numpy as np
  
  # torch.tensor(data=, dtype=): Create a tensor based on specified data or specified element type
  # data: input data
  # dtype: element data type
  def dm01():
	list1 = [[1., 2, 3], [4, 5, 6]]  # The created tensor will be float32 by default
	int1 = 10
	# NumPy array default type is float64, so the created tensor will be float64
	n1 = np.array([[1., 2., 3.], [4., 5., 6.]])
	
	t1 = torch.tensor(data=list1)
	t2 = torch.tensor(data=int1)
	t3 = torch.tensor(data=n1)
	
	print('t1 value ->', t1)
	print('t1 type ->', type(t1))
	print('t1 element type ->', t1.dtype)
	
	print('t2 value ->', t2)
	print('t2 type ->', type(t2))
	
	print('t3 value ->', t3)
	print('t3 type ->', type(t3))


  # torch.Tensor(data=, size=): Create a tensor based on specified data or specified shape
  # data: input data
  # size: shape (tuple) -> (dim0, dim1, ...)
  # The number of elements in the tuple determines the number of dimensions,
  # and each value corresponds to the size along that dimension.
  def dm02():
	# Specify data
	t1 = torch.Tensor(data=[[1.1, 1.2, 1.3], [2.2, 2.3, 2.4]])
	print('t1 value ->', t1)
	print('t1 type ->', type(t1))
	
	# Specify shape
	t2 = torch.Tensor(size=(2, 3))
	print('t2 value ->', t2)
	print('t2 type ->', type(t2))


  # torch.IntTensor(data) / LongTensor() / FloatTensor() / DoubleTensor():
  # Create tensors with specified data types
  # data: input data
  def dm03():
	# If the element type does not match the specified type,
	# it will be automatically converted
	t1 = torch.IntTensor([[1.1, 2, 3.7], [4, 5, 6]])
	t2 = torch.FloatTensor([[1.1, 2, 3.7], [4, 5, 6]])
	
	print('t1 value ->', t1)
	print('t1 type ->', type(t1))
	print('t1 element type ->', t1.dtype)
	
	print('t2 value ->', t2)
	print('t2 type ->', type(t2))
	print('t2 element type ->', t2.dtype)


  if __name__ == '__main__':
	dm01()
	# dm02()
	# dm03()

### 3.3 Linear and Random Tensors

- Linear Tensor

  - torch.arange()
  - torch.linspace()

- Random Tensor

  - torch.rand()/randn()
  - torch.randint()
  - torch.initial_seed()
  - torch.manual_seed()

  ```python
  import torch

  # torch.arange(): Create linear tensor with step size (left-closed, right-open)
  # torch.linspace(): Create linear tensor with specified number of elements (left-closed, right-closed)
  def dm01():
    t1 = torch.arange(start=0, end=10, step=2)
    print('t1 value ->', t1)

    t2 = torch.linspace(start=0, end=9, steps=9)
    print('t2 value ->', t2)


  # torch.rand() / randn(): Create random float tensors
  # torch.randint(): Create random integer tensor (left-closed, right-open)
  # torch.initial_seed(): View random seed
  # torch.manual_seed(): Set random seed
  def dm02():
    t1 = torch.rand(size=(5, 4))
    print('t1 value ->', t1)
    print('t1 dtype ->', t1.dtype)

    torch.manual_seed(seed=66)
    t2 = torch.randint(low=0, high=10, size=(2, 3))
    print('t2 value ->', t2)
    print('t2 dtype ->', t2.dtype)


  if __name__ == '__main__':
    dm02()
  ```

### 3.4 0/1/Specified Value Tensors

- torch.ones/zeros/full(size=[, fill_value=])

- torch.ones_like/zeros_like/full_like(input=tensor[, fill_value=])

  ```python
  import torch
  
  
  # torch.ones(size=): 根据形状创建全1张量
  # torch.ones_like(input=): 根据指定张量的形状创建全1张量
  def dm01():
  	t1 = torch.ones(size=(2, 3))
  	print('t1的值是->', t1)
  	print('t1的形状是->', t1.shape)
  	print('t1的元素类型是->', t1.dtype)
  	# 形状: (5, )
  	t2 = torch.tensor(data=[1, 2, 3, 4, 5])
  	t3 = torch.ones_like(input=t2)
  	print('t2的形状是->', t2.shape)
  	print('t3的值是->', t3)
  	print('t3的形状是->', t3.shape)
  
  
  # torch.zeros(size=): 根据形状创建全0张量
  # torch.zeros_like(input=): 根据指定张量的形状创建全0张量
  def dm02():
  	t1 = torch.zeros(size=(2, 3))
  	print('t1的值是->', t1)
  	print('t1的形状是->', t1.shape)
  	print('t1的元素类型是->', t1.dtype)
  	# 形状: (5, )
  	t2 = torch.tensor(data=[1, 2, 3, 4, 5])
  	t3 = torch.zeros_like(input=t2)
  	print('t2的形状是->', t2.shape)
  	print('t3的值是->', t3)
  	print('t3的形状是->', t3.shape)
  
  
  # torch.full(size=, fill_value=): 根据形状和指定值创建指定值的张量
  # torch.full_like(input=, fill_value=): 根据指定张量形状和指定值创建指定值的张量
  def dm03():
  	t1 = torch.full(size=(2, 3, 4), fill_value=10)
  	t2 = torch.tensor(data=[[1, 2], [3, 4]])
  	t3 = torch.full_like(input=t2, fill_value=100)
  	print('t1的值是->', t1)
  	print('t1的形状是->', t1.shape)
  	print('t3的值是->', t3)
  	print('t3的形状是->', t3.shape)
  
  if __name__ == '__main__':
  	# dm01()
  	# dm02()
  	dm03()
  ```

### 3.5 Specifying Tensor Data Types

- tensor.type(dtype=)

- tensor.half()/float()/double()/short()/int()/long()

  ```python
  import torch
  
  
  # torch.tensor(data=, dtype=):
  # dtype: specify the element data type (default floating type is float32)

  # tensor.type(dtype=): change the data type of tensor elements
  # torch.float32
  # torch.FloatTensor
  # torch.cuda.FloatTensor
  def dm01():
    t1 = torch.tensor(data=[[1., 2., 3.], [4., 5., 6.]], dtype=torch.float16)
    print('t1 dtype ->', t1.dtype)

    # convert to float32
    t2 = t1.type(dtype=torch.FloatTensor)
    t3 = t1.type(dtype=torch.int64)

    print('t2 dtype ->', t2.dtype)
    print('t3 dtype ->', t3.dtype)


  # tensor.half() / float() / double() / short() / int() / long()
  def dm02():
    t1 = torch.tensor(data=[1, 2])
    print('t1 dtype ->', t1.dtype)
    # t2 = t1.half()
    t2 = t1.int()
    print(t2)
    print('t2 dtype ->', t2.dtype)

  if __name__ == '__main__':
    # dm01()
    dm02()
    ```

## 4 Tensor Type Conversion

### 4.1 Converting Tensor to NumPy Array

```python
import torch
import numpy as np


# Convert tensor to NumPy array
# tensor.numpy(): shares memory; modifying one will affect the other.
# Use .copy() to avoid shared memory.
def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print('t1 ->', t1)
    # Convert to NumPy array
    # n1 = t1.numpy()
    n1 = t1.numpy().copy()
    print('n1 ->', n1)
    print('n1 type ->', type(n1))
    # Modify the first element of n1
    # [0][0] -> element in the first row and first column
    n1[0][0] = 100
    print('n1 after modification ->', n1)
    print('t1 ->', t1)
	```

### 4.2 Converting NumPy Array to Tensor

```python
# Convert NumPy array to tensor
# torch.from_numpy(ndarray): shares memory with the ndarray
# torch.tensor(data=ndarray): does NOT share memory
def dm02():
    n1 = np.array([[1, 2, 3], [4, 5, 6]])
    # Convert to tensor
    # Shared memory
    t1 = torch.from_numpy(n1)
    # Not shared memory
    # t1 = torch.from_numpy(n1.copy())
    # t1 = torch.tensor(data=n1)
    print('t1 ->', t1)
    print('t1 type ->', type(t1))
    # Modify tensor element
    t1[0][0] = 8888
    print('t1 after modification ->', t1)
    print('n1 ->', n1)
```

### 4.3 Extracting Scalar Value from a Tensor

```python
import torch


# tensor.item(): Extract the value from a single-element tensor.
# The tensor can be a scalar tensor, 1D tensor, 2D tensor, etc.,
# as long as it contains only one element.
def dm01():
    # Convert a numerical value into a tensor
    # Scalar tensor
    t1 = torch.tensor(data=10)
    # 1D tensor
    # t1 = torch.tensor(data=[10])
    # 2D tensor
    # t1 = torch.tensor(data=[[10]])   
    print('t1 ->', t1)
    print('t1 shape ->', t1.shape)   
    # Convert single-element tensor to a Python value
    print('t1.item() ->', t1.item())


if __name__ == '__main__':
	dm01()
```

## 5 Tensor Numerical Operations

### 5.1 Basic Operations

- `+` `-` `*` `/` `-`

- tensor/torch.add() sub() mul() div() neg()

- `tensor/torch.add_()` `sub_()` `mul_()` `div_()` `neg_()`

  ```python
  import torch
  
  
  # Operations: tensor-scalar operations and tensor-tensor operations
  # + - * / -
  # add(other=) sub() mul() div() neg()  → do NOT modify the original tensor
  # add_() sub_() mul_() div_() neg_() → modify the original tensor (in-place operations)

  def dm01():
    # Create a tensor
    t1 = torch.tensor(data=[1, 2, 3, 4])

    # Tensor and scalar operation
    t2 = t1 + 10
    print('t2 ->', t2)

    # Tensor and tensor operation (element-wise computation)
    t3 = t1 + t2
    print('t3 ->', t3)

    # add() does NOT modify the original tensor
    t1.add(other=100)
    t4 = torch.add(input=t1, other=100)
    print('t4 ->', t4)

    # neg_() modifies the original tensor (in-place), applies negative sign
    t5 = t1.neg_()
    print('t1 ->', t1)
    print('t5 ->', t5)
  	t5 = t1.neg_()
  	print('t1->', t1)
  	print('t5->', t5)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

### 5.2 Element-wise Multiplication (Hadamard Product)

- Multiply corresponding elements
- Usually requires same shape

  ```python
  import torch
  
  
  # Element-wise multiplication (also called Hadamard product):
  # Tensor element-level multiplication.
  # Multiply elements at corresponding positions.
  # Usually requires the two tensors to have the same shape. 

  def dm01():
  	# t1 = torch.tensor(data=[[1, 2], [3, 4]])
  	# (2, )
  	t1 = torch.tensor(data=[1, 2])
  	# (2, 2)
  	t2 = torch.tensor(data=[[5, 6], [7, 8]])
  	t3 = t1 * t2
  	print('t3->', t3)
  	t4 = torch.mul(input=t1, other=t2)
  	print('t4->', t4)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

### 5.3 Matrix Multiplication

- Multiply the row data of the first matrix by the column data of the second matrix.

  ```python
  import torch
  
  
  # Matrix Multiplication: (n, m) * (m, p) = (n, p)  the row of first matrix multiply the column of seconf matrix  @  torch.matmul(input=, ohter=)
  def dm01():
  	# (2, 2)
  	t1 = torch.tensor(data=[[1, 2],
  							[3, 4]])
  	# (2, 3)
  	t2 = torch.tensor(data=[[5, 6, 7],
  							[8, 9, 10]])
  
  	# @
  	t3 = t1 @ t2
  	print('t3->', t3)
  	# torch.matmul(): Different shapes are acceptable provided the subsequent dimensions satisfy the rules of matrix multiplication.
  	t4 = torch.matmul(input=t1, other=t2)
  	print('t4->', t4)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

## 6 Tensor Mathematical Functions

- mean()

- sum()

- min()/max()

- dim: compute along specific dimensions

- exp(): exponential

- sqrt(): square root

- pow(): power

- log()/log2()/log10(): logrithm

  ```python
  import torch
  
  
  def dm01():
  	# create tensor
  	t1 = torch.tensor(data=[[1., 2, 3, 4],
  							[5, 6, 7, 8]])
  
  	# dim=0 by column
  	# dim=1 by row
  	# Mean
  	print('Mean for all value->', t1.mean())
  	print('Mean for each column->', t1.mean(dim=0))
  	print('Mean for each row->', t1.mean(dim=1))
  	# Sum
  	print('Sum for all value-->', t1.sum())
  	print('Sum for each column->', t1.sum(dim=0))
  	print('Sum for each row->', t1.sum(dim=1))
  	# sqrt: Square root
  	print('Take the square root of all values->', t1.sqrt())
  	# pow: Power  x^n
  	print('Power->',torch.pow(input=t1, exponent=2))
  	# exp: Exponential e^x  
  	print('Exponential->', torch.exp(input=t1))
  	# log: logrithm  log(x)->e:base  log2()  log10()
  	print('base-e logarithm->', torch.log(input=t1))
  	print('base-2 logarithm->', t1.log2())
  	print('base-10 logarithm->', t1.log10())
  
  
  if __name__ == '__main__':
  	dm01()
  ```

  



















