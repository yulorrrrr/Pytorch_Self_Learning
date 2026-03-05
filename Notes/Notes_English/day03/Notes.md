# Notes 

## 1 Building a Linear Regression Model with PyTorch

### 1.1 Create Dataset

```python
import torch
from torch.utils.data import TensorDataset  # Create tensor dataset object for x and y
from torch.utils.data import DataLoader  # Create data loader
import torch.nn as nn  # Loss function and regression model
from torch.optim import SGD  # Stochastic Gradient Descent optimizer (compute gradient using one sample)
from sklearn.datasets import make_regression  # Generate random samples (not used in production)
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Display Chinese labels correctly
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly



# todo: 1 - Create linear regression samples x, y, coefficient (w), and bias (b)
def create_datasets():
    x, y, coef = make_regression(
        n_samples=100,  # number of samples
        n_features=1,   # number of features
        noise=10,       # standard deviation (noise, dispersion level)
        coef=True,      # return coefficient (w)
        bias=14.5,      # intercept (b)
        random_state=0
    )

    # Convert arrays to tensors
    x = torch.tensor(data=x)
    y = torch.tensor(data=y)
	# print('x->', x)
	# print('y->', y)
	# print('coef->', coef)
	return x, y, coef

if __name__ == '__main__':
	x, y, coef = create_datasets()
```

### 1.2 Training Model

```python
# todo: 2 - Model training
def train(x, y, coef):
	# Create tensor dataset object
	datasets = TensorDataset(x, y)
	print('datasets->', datasets)
	# Create data loader
    # dataset: tensor dataset object
    # batch_size: number of samples per batch
    # shuffle: whether to shuffle samples
	dataloader = DataLoader(dataset=datasets, batch_size=16, shuffle=True)
	print('dataloader->', dataloader)
	# for batch in dataloader:  # Each time traversing to retrieve each batch sample
	# 	print('batch->', batch)  # [xTensor object, yTensor object]
	# 	break
	# Create initial regression model (randomly generate w and b, dtype=float32)
    # in_features: number of input features
    # out_features: number of output features
	model = nn.Linear(in_features=1, out_features=1)
	print('model->', model)
	# Get model parameters
	print('model.weight->', model.weight)
	print('model.bias->', model.bias)
	print('model.parameters()->', list(model.parameters()))
	# Create loss function (calculate loss value)
	criterion = nn.MSELoss()
	# Create loss function (calculate loss value)
	optimizer = SGD(params=model.parameters(), lr=0.01)
	# # Define variables to receive training iterations, loss values, and training sample counts
	epochs = 100
	loss_list = []  # Store the average loss value for each training session
	total_loss = 0.0
	train_samples = 0
	for epoch in range(epochs):  # train 100 time
        # Mini-batch SGD training
		for train_x, train_y in dataloader:
			# Model prediction
			# train_x->float64
			# w->float32
			y_pred = model(train_x.type(dtype=torch.float32))  # y=w*x+b
			print('y_pred->', y_pred)
			# Calculate the loss value
			# print('train_y->', train_y)
			# y_pred: 2 dim tensor
			# train_y: modify 1 dim tensor to 2 dim, nrow 1column
			# An error may occur; modify the shape.
			# Modify the element type of `train_y` to match that of `y_pred`, otherwise an error 				will occur.
			loss = criterion(y_pred, train_y.reshape(shape=(-1, 1)).type(dtype=torch.float32))
			print('loss->', loss)
			# Get the scalar value from the loss tensor using item()
			# Accumulate the total MSE over n batches
			total_loss += loss.item()
			# compute the number of batches
			train_samples += 1
			# Zeroing gradients
			optimizer.zero_grad()
			# Clear gradients
			loss.backward()
			# Gradient update (update w and b)
			# step() is equivalent to: w = w - lr * grad
			optimizer.step()
		# Save the average loss of each epoch into the loss_list
		loss_list.append(total_loss / train_samples)
		print('Average training loss per epoch->', total_loss / train_samples)
	print('loss_list->', loss_list)
	print('w->', model.weight)
	print('b->', model.bias)
    
    # Plot the loss curve over epochs
	plt.plot(range(epochs), loss_list)
	plt.title('Loss Curve')
	plt.grid()
	plt.show()

	# Plot comparison between predicted values and ground truth
    # Plot sample point distribution
	plt.scatter(x, y)
	# Get 1000 sample points
	# x = torch.linspace(start=x.min(), end=x.max(), steps=1000)
	# Compute predicted values from trained model
	y1 = torch.tensor(data=[v * model.weight + model.bias for v in x])
	# Compute ground truth values
	y2 = torch.tensor(data=[v * coef + 14.5 for v in x])
	plt.plot(x, y1, label='训练')
	plt.plot(x, y2, label='真实')
	plt.legend()
	plt.grid()
	plt.show()


if __name__ == '__main__':
	x, y, coef = create_datasets()
	train(x, y, coef)
```

## 2 Introduction to Artificial Neural Networks

### 2.1 What is an Artificial Neural Network

- A computational model inspired by biological neural networks
- ANN (Artificial Neural Network) → NN (Neural Network)


### 2.2 How to Construct an Artificial Neural Network

> A neural network consists of three types of layers, each composed of multiple neurons

- Input layer: inputs feature values (one layer)
- Hidden layer: extracts complex features (can have multiple layers)
- Output layer: outputs predicted y value

### 2.3 Internal State Value and Activation Value

> How a neuron works

- Internal state value (weighted sum)
  - `z=w1*x1+w2*x2+...+b`
- Activation value
  - `a=f(z)`

### 3 Introduction to Activation Functions

### 3.1 Purpose of Activation Functions

- Introduce non-linearity into neural networks
- In real-world problems, data is often not linearly separable

### 3.2 Common Activation Functions

- sigmoid Activation Function

  - Output range: [0, 1]‘
  - Only positive signals (no negative signal learning)
  - Weighted sum should be in [-6, 6] to avoid saturation
  - Derivative range: [0, 0.25]
  - Small derivative → vanishing gradient (0.25 * 0.25 * 0.25 ...)
  - Typically used in binary classification output layer
  - Can be used in shallow networks (< 5 layers)

  ```python
  # Sigmoid activation value: torch.sigmoid(x)

  import torch
  import matplotlib.pyplot as plt

  plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to correctly display Chinese labels
  plt.rcParams['axes.unicode_minus'] = False  # Used to correctly display minus signs


  def dm01():
    # Create x values; the output of the linear model serves as the input to the activation     	function
    x = torch.linspace(-20, 20, 1000)

    # Compute activation values
    y = torch.sigmoid(input=x)

    # Create figure and axes objects
    _, axes = plt.subplots(1, 2)  # One row and two columns, draw two subplots

    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title('Sigmoid Activation Function')

    # Create x values with automatic differentiation enabled
    # The output of the linear model serves as the input to the activation function
    x = torch.linspace(-20, 20, 1000, requires_grad=True)

    torch.sigmoid(input=x).sum().backward()

    axes[1].plot(x.detach().numpy(), x.grad)
    axes[1].grid()
    axes[1].set_title('Sigmoid Activation Function')

    plt.show()


  if __name__ == '__main__':
    dm01()
  ```

- Tanh Activation Function
  - Output range: [-1, 1]
  - Can learn both positive and negative signals
  - Derivative range: [0, 1]
  - Converges faster than sigmoid
  - Still may suffer gradient vanishing if |z| > 3
  - Can be used in hidden layers (not first choice)

  ```python
  # Tanh activation value: torch.tanh(x)

  import torch
  import matplotlib.pyplot as plt
  from torch.nn import functional as F
  # F.sigmoid()
  # F.tanh()

  plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to correctly display Chinese labels
  plt.rcParams['axes.unicode_minus'] = False  # Used to correctly display minus signs


  def dm01():
    # Create x values; the output of the linear model serves as the input to the activation   		function
    x = torch.linspace(-20, 20, 1000)

    # Compute activation values
    y = torch.tanh(input=x)

    # Create figure and axes objects
    _, axes = plt.subplots(1, 2)  # One row and two columns, draw two subplots

    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title('Tanh Activation Function')

    # Create x values with automatic differentiation enabled
    # The output of the linear model serves as the input to the activation function
  	x = torch.linspace(-20, 20, 1000, requires_grad=True)
  	torch.tanh(input=x).sum().backward()
  	axes[1].plot(x.detach().numpy(), x.grad)
  	axes[1].grid()
  	axes[1].set_title('Tanh Activation Function')
  	plt.show()
  
  
  if __name__ == '__main__':
  	dm01()
  ```

- ReLU Activation Function
  - Output range: [0, x]
  - Derivative is 0 or 1
  - No vanishing gradient when z > 0
  - May cause dead neurons when z < 0
  - Preferred activation function
  - Low computational cost

  ```python
  # ReLU activation value: torch.relu(x)

  import torch
  import matplotlib.pyplot as plt
  from torch.nn import functional as F
  # F.sigmoid()
  # F.tanh()

  plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to correctly display Chinese labels
  plt.rcParams['axes.unicode_minus'] = False  # Used to correctly display minus signs


  def dm01():
    # Create x values; the output of the linear model serves as the input to the activation 		function
    x = torch.linspace(-20, 20, 1000)

    # Compute activation values
    y = torch.relu(input=x)

    # torch.leaky_relu()
    # torch.prelu()

    # Create figure and axes objects
    axes = plt.subplots(1, 2)  # One row and two columns, draw two subplots
    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title('ReLU Activation Function')

    # Create x values with automatic differentiation enabled
    # The output of the linear model serves as the input to the activation function
  	x = torch.linspace(-20, 20, 1000, requires_grad=True)
  	torch.relu(input=x).sum().backward()
  	axes[1].plot(x.detach().numpy(), x.grad)
  	axes[1].grid()
  	axes[1].set_title('relu activation function')
  	plt.show()
  
  
  if __name__ == '__main__':
  	dm01()
  ```

- Softmax Activation Function
  - Used in multi-class classification output layer
  - Converts weighted sums into probabilities

  ```python
  import torch
  import pandas as pd
  
  
  def dm01():
  	# Create the weighted sum values of the output layer
    y = torch.tensor(data=[
    [0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75],
    [0.2, 0.02, 0.15, 3.75, 1.3, 0.5, 0.06, 1.1, 0.05, 0.15]])

  # Convert the weighted sums into probability values using the softmax activation function
  # Compute along axis 1 (column-wise)
  # y_softmax = torch.softmax(input=y, dim=-1)
  	y_softmax = torch.softmax(input=y, dim=1)
  	print('y_softmax->', y_softmax)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

### 3.6 How to Choose Activation Functions

- Hidden Layers
  - Prefer ReLU
  - Next: Leaky ReLU / PReLU
  - Avoid sigmoid
  - Tanh for shallow networks

- Output Layer
  - Binary classification → Sigmoid
  - Multi-class classification → Softmax
  - Regression → Identity (no activation)

## 4 Parameter Initialization

### 4.1 Purpose of Parameter Initialization

- Parameters → w and b
- Assign values to w and b when creating the initial model
- Proper initialization:
  - Ensures weighted sum falls within effective activation range
  - Speeds up convergence
  - Enables learning of diverse features

### 4.2 Common Parameter Initialization Methods

- Random initialization
- All zeros / ones initialization
- Constant initialization
- Kaiming initialization
- Xavier initialization

  ```python
  import torch
  import torch.nn as nn
  
  
  # Random parameter initialization
   def dm01():
    # Create linear layer objects and initialize their weights
    # in_features: number of input neurons
    # out_features: number of output neurons
    linear1 = nn.Linear(in_features=5, out_features=8)
    linear2 = nn.Linear(in_features=8, out_features=10)

    # Uniform distribution initialization
    # By default, values are sampled uniformly from (0, 1)
    # The range can be adjusted using parameters a and b
    nn.init.uniform_(linear1.weight)
    nn.init.uniform_(
        linear1.weight,
        a=-1 / torch.sqrt(torch.tensor(5.0)),
        b=1 / torch.sqrt(torch.tensor(5.0))
    )
    nn.init.uniform_(linear1.bias)

    print(linear1.weight)
    print(linear1.bias)


  # Normal distribution parameter initialization
  def dm02():
    # Create linear layer objects and initialize their weights
    # in_features: number of input neurons
    # out_features: number of output neurons
    linear1 = nn.Linear(in_features=5, out_features=8)
    linear2 = nn.Linear(in_features=8, out_features=10)

    # Normal distribution initialization
    nn.init.normal_(linear1.weight)
    nn.init.normal_(linear1.bias)

    print(linear1.weight)
    print(linear1.bias)


  # nn.init.zeros_()          # All-zero initialization
  # nn.init.ones_()           # All-one initialization
  # nn.init.constant_(val=0.1) # Constant value initialization
  # nn.init.kaiming_uniform_() # Kaiming uniform initialization
  # nn.init.kaiming_normal_()  # Kaiming normal initialization
  # nn.init.xavier_uniform_()  # Xavier uniform initialization
  # nn.init.xavier_normal_()   # Xavier normal initialization
  
  
  if __name__ == '__main__':
  	dm01()
  	dm02()
  ```

### 4.3 How to Choose Initialization Methods

- Shallow networks → random initialization
- Deep networks → choose based on activation function:
  - tanh → Xavier initialization

  - ReLU → Kaiming initialization
