# Notes

##  Neural Network Construction and Parameter Calculation

###  Neural Network Construction and Parameter Calculation

```python
import torch
import torch.nn as nn  # Linear models and initialization methods


# todo:1 - Create a class that inherits from nn.Module
class ModelDemo(nn.Module):
	# todo:2 - Define the __init__ constructor to build the neural network
	def __init__(self):
		# todo:2-1 Call the parent class __init__ method
		super().__init__()

		# todo:2-2 Create hidden layers and output layer, define attributes
		# in_features: number of input features (number of neurons in previous layer)
		# out_features: number of output features (number of neurons in current layer)
		self.linear1 = nn.Linear(in_features=3, out_features=3)
		self.linear2 = nn.Linear(in_features=3, out_features=2)
		self.output = nn.Linear(in_features=2, out_features=2)
		# todo:2-3 Initialize parameters of hidden layers
		# self.linear1.weight: calling object attributes inside the class
		nn.init.xavier_normal_(tensor=self.linear1.weight)
		nn.init.zeros_(tensor=self.linear1.bias)
		nn.init.kaiming_normal_(tensor=self.linear2.weight, nonlinearity='relu')
		nn.init.zeros_(tensor=self.linear2.bias)

	# todo:3 Define forward propagation method forward (method name is fixed) to get predicted y
	def forward(self, x):  # x -> input sample features
		# todo:3-1 First layer computation
		# weighted sum + activation function
		x = torch.sigmoid(input=self.linear1(x))

		# todo:3-2 Second layer computation
		x = torch.relu(input=self.linear2(x))

		# todo:3-3 Output layer computation (assume multi-class classification)
		# dim=-1: compute row-wise (one sample at a time)
		x = torch.softmax(input=self.output(x), dim=-1)

		# return predicted values
		return x


if __name__ == '__main__':
	# Create neural network model object
	my_model = ModelDemo()
```

### Calculating and Viewing Model Parameters

```python
# Create model prediction function
def train():

	# todo:1 Create neural network model object
	my_model = ModelDemo()

	print('my_model->', my_model)

	# todo:2 Construct dataset samples (randomly generated)
	data = torch.randn(size=(5, 3))

	print('data->', data)
	print('data.shape->', data.shape)

	# todo:3 Call neural network model for prediction
	output = my_model(data)

	print('output->', output)
	print('output.shape->', output.shape)

	# todo:4 Calculate and view model parameters
	print(('====================Calculate and View Model Parameters==================='))

	# input_size: number of features per sample
	# batch_size: number of samples in batch training

	# summary(model=my_model, input_size=(3,), batch_size=5)
	summary(model=my_model, input_size=(5, 3))

	for name, param in my_model.named_parameters():
		print('name->', name)
		print('param->', param)


if __name__ == '__main__':
	train()
```

## Loss Functions

### What is a Loss Function

- A function used to measure the quality of model parameters (evaluate the model)

- Compares the difference between predicted values and true values

- Guides parameter updates by computing gradients using gradient descent loss.backward()

###  Classification Loss Functions

#### Multi-class Classification Loss

- Suitable for multi-class classification problems
- After computing the loss, gradient descent and backpropagation are used to update parameters
- nn.CrossEntropyLoss()

```python
# Suitable for multi-class classification

import torch
import torch.nn as nn


def dm01():
	# Manually create ground truth y values
	# y_true = torch.tensor(data=[[0,1,0],[1,0,0]], dtype=torch.float32)
	y_true = torch.tensor(data=[1,2])
	print('y_true->', y_true.dtype)

	# Manually create predicted y values -> model predictions
	y_pred = torch.tensor(
		data=[[0.1,0.8,0.1],[0.7,0.2,0.1]],
		requires_grad=True,
		dtype=torch.float32
	)

	# Create cross entropy loss object
	# reduction: method for computing loss value, default is mean
	criterion = nn.CrossEntropyLoss(reduction='sum')

	# Compute loss value
	loss = criterion(y_pred, y_true)
	print('loss->', loss)


if __name__ == '__main__':
	dm01()
```

####  Binary Classification Loss

- Suitable for binary classification
- nn.BCELoss()

```python
# Suitable for binary classification

import torch
import torch.nn as nn


def dm01():
	# Manually create the ground truth y values of the samples
	y_true = torch.tensor(data=[0, 1, 0], dtype=torch.float32)
	print('y_true->', y_true.dtype)

	# Manually create the predicted y values -> model prediction results
	# 0.6901, 0.5459, 0.2469 -> activation values from the sigmoid function
	y_pred = torch.tensor(data=[0.6901, 0.5459, 0.2469], requires_grad=True, dtype=torch.float32)

	# Create binary cross-entropy loss object
	# reduction: method used to compute the loss value, default is mean (average loss)
	criterion = nn.BCELoss()

	# Use the loss object to compute the loss value
	# predicted y, true y
	loss = criterion(y_pred, y_true)

	print('loss->', loss)
	print('loss->', loss.requires_grad)


if __name__ == '__main__':
	dm01()
```

### Regression Loss Functions

####  MAE Loss

- loss = (|y_true - y_pred| + ... ) / n
- Derivative is -1 or 1
- Not differentiable at 0

```python
# Suitable for regression tasks
# MAE: derivative is -1 or 1; not differentiable at 0, usually taken as 0 in practice

import torch
import torch.nn as nn


def dm01():

	# Manually create the ground truth y values of the samples
	y_true = torch.tensor(data=[1.2, 1.5, 2.0], dtype=torch.float32)
	print('y_true->', y_true.dtype)

	# Manually create the predicted y values -> model predictions
	# 0.6901, 0.5459, 0.2469 -> activation values from the sigmoid function
	y_pred = torch.tensor(data=[1.3, 1.7, 2.0], requires_grad=True, dtype=torch.float32)

	# Create MAE loss object for regression tasks
	# reduction: method for computing the loss value, default is mean (average loss)
	criterion = nn.L1Loss()

	# Use the loss object to compute the loss value
	# predicted y, true y
	loss = criterion(y_pred, y_true)
	print('loss->', loss)
	print('loss->', loss.requires_grad)


if __name__ == '__main__':
	dm01()
```

#### MSE Loss

- If the dataset contains outliers, errors may be amplified
- May cause gradient explosion
- Differentiable everywhere

```python
# Suitable for regression tasks. If the dataset contains outliers,
# the error may be amplified, which could lead to gradient explosion.
# MSE: differentiable at any point, and the closer to the minimum,
# the smaller the gradient.

import torch
import torch.nn as nn


def dm01():
	# Manually create the ground truth y values of the samples
	y_true = torch.tensor(data=[1.2, 1.5, 2.0], dtype=torch.float32)
	print('y_true->', y_true.dtype)

	# Manually create the predicted y values -> model predictions
	# 0.6901, 0.5459, 0.2469 -> activation values from the sigmoid function
	y_pred = torch.tensor(data=[1.3, 1.7, 2.0], requires_grad=True, dtype=torch.float32)

	# Create MSE loss object for regression tasks
	# reduction: method for computing the loss value, default is mean (average loss)
	criterion = nn.MSELoss()

	# Use the loss object to compute the loss value
	# predicted y, true y
	loss = criterion(y_pred, y_true)
	print('loss->', loss)
	print('loss->', loss.requires_grad)


if __name__ == '__main__':
	dm01()
```

#### Smooth L1Loss

- Combines advantages of MAE and MSE
- When error >1 → MAE
- When error within [-1,1] → MSE
- Not sensitive to outliers
- Differentiable everywhere

```python
# Suitable for regression tasks, not sensitive to outliers
# Smooth L1: differentiable at any point, the closer to the minimum, the smaller the gradient

import torch
import torch.nn as nn


def dm01():

	# Manually create the ground truth y values of the samples
	y_true = torch.tensor(data=[1.2, 1.5, 2.0], dtype=torch.float32)
	print('y_true->', y_true.dtype)

	# Manually create the predicted y values -> model predictions
	# 0.6901, 0.5459, 0.2469 -> activation values from the sigmoid function
	y_pred = torch.tensor(data=[1.3, 1.7, 2.0], requires_grad=True, dtype=torch.float32)

	# Create Smooth L1 loss object for regression tasks
	# reduction: method for computing the loss value, default is mean (average loss)
	criterion = nn.SmoothL1Loss()

	# Use the loss object to compute the loss value
	# predicted y, true y
	loss = criterion(y_pred, y_true)
	print('loss->', loss)
	print('loss->', loss.requires_grad)


if __name__ == '__main__':
	dm01()
```

##  Neural Network Optimization Methods

### Review of Gradient Descent

- W = W - lr * grad

- Gradient descent is a strategy to find optimal network parameters.
  - BGD: uses all samples to compute gradient (high cost)
  - SGD: uses one random sample (unstable gradient)
  - Mini-batch: uses a batch of samples (balanced approach)

### Backpropagation (BP)

- Backpropagation computes gradients of loss with respect to parameters.
- Combine the gradient descent algorithm to update parameters: W₁ = W₀ − LR × gradient
- Here, W₀ and LR are known, and the gradient is computed using the backpropagation (BP) algorithm.

### Optimization Algorithms

> Avoid encountering saddle points and local minima situations

####  Exponential Moving Average

- $$S_t​$$ = $(1-β)Y_t​$ + $$βS_{t-1}​$$

- S100 = `0.1*Y100 + 0.1*0.9*Y99 + 0.1*0.9*0.9*Y98 + ....`
- The greater β, the less the current moment's true value is affected; typically 0.9.


#### Momentum

- $$S_t$$ = $(1-β)g_t$ + $$βS_{t-1}$$

  ```python
  # Momentum computes exponential moving average of gradients.
  import torch
  from torch import optim
  
  
  def dm01():
  	# todo: 1 - initialize weight parameter
  	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
	# custom loss function
    # in real tasks we use different loss functions depending on the task
    # e.g. Cross-Entropy loss / MSE loss ...
  	loss = ((w ** 2) / 2.0).sum()
  	# todo: 2 - create optimizer object (SGD with momentum)
    # momentum: momentum method, usually 0.9 or 0.99
  	optimizer = optim.SGD([w], lr=0.01, momentum=0.9)
  	# todo: 3-compute gradient
  	optimizer.zero_grad()
  	loss.sum().backward()
  	# todo: 4-update weight parameter (gradient update)
    optimizer.step()
  	optimizer.step()
  	print('w.grad->', w.grad)
      # second calculation
      loss = ((w ** 2) / 2.0).sum()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      print('w.grad->', w.grad)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

####  AdaGrad

- Learning rate decreases during training
- Large at beginning, smaller later

  ```python
  # AdaGrad optimization method adjusts the learning rate.
  # As the number of training iterations increases, the learning rate becomes smaller.
  # The learning rate is relatively large at the beginning.

  import torch
  from torch import optim

def dm01():

	# todo: 1 - initialize weight parameter
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

	loss = ((w ** 2) / 2.0).sum()

	# todo: 2 - create optimizer object (Adagrad)
	optimizer = optim.Adagrad([w], lr=0.01)

	# todo: 3 - compute gradient
	optimizer.zero_grad()
	loss.sum().backward()

	# todo: 4 - update weight parameter (gradient update)
	optimizer.step()

	print('w.grad->', w.grad, w)

	# second calculation
	loss = ((w ** 2) / 2.0).sum()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('w.grad->', w.grad, w)

	# third calculation
  	loss = ((w ** 2) / 2.0).sum()
  	optimizer.zero_grad()
  	loss.backward()
  	optimizer.step()
  	print('w.grad->', w.grad, w)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

#### RMSProp

- Improvement of AdaGrad
- Uses exponential moving average of gradients.

  ```python
  # RMSprop optimization method adjusts the learning rate.
  # It improves the AdaGrad method by using the exponential moving average of gradients
  # instead of the accumulated historical gradients.
  # This avoids the learning rate decreasing too quickly,
  # which would slow down the model convergence in later training.

  import torch
  from torch import optim

  def dm01():

	# todo: 1 - initialize weight parameter
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

	loss = ((w ** 2) / 2.0).sum()

	# todo: 2 - create optimizer object (RMSprop)
	optimizer = optim.RMSprop([w], lr=0.01, alpha=0.9)

	# todo: 3 - compute gradient
	optimizer.zero_grad()
	loss.sum().backward()

	# todo: 4 - update weight parameter (gradient update)
	optimizer.step()

	print('w.grad->', w.grad, w)

	# second calculation
	loss = ((w ** 2) / 2.0).sum()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('w.grad->', w.grad, w)

	# third calculation
  	loss = ((w ** 2) / 2.0).sum()
  	optimizer.zero_grad()
  	loss.backward()
  	optimizer.step()
  	print('w.grad->', w.grad, w)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

#### Adam

- Combines Momentum + RMSProp

  ```python
  # Adam optimizes both the learning rate and the gradient
  # Adam = RMSprop + Momentum

  import torch
  from torch import optim

  def dm01():
	# todo: 1 - initialize weight parameter
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

	loss = ((w ** 2) / 2.0).sum()

	# todo: 2 - create optimizer object
	# betas: two β values, passed as a tuple
	optimizer = optim.Adam([w], lr=0.01, betas=(0.9, 0.99))

	# todo: 3 - compute gradient
	optimizer.zero_grad()
	loss.sum().backward()

	# todo: 4 - update weight parameter (gradient update)
	optimizer.step()

	print('w.grad->', w.grad, w)

	# second calculation
	loss = ((w ** 2) / 2.0).sum()
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('w.grad->', w.grad, w)

	# third calculation
  	loss = ((w ** 2) / 2.0).sum()
  	optimizer.zero_grad()
  	loss.backward()
  	optimizer.step()
  	print('w.grad->', w.grad, w)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

####  Choosing Optimization Methods

- SGD / Momentum → simple tasks or shallow networks
- Adam → complex tasks or large datasets
- AdaGrad / RMSprop → NLP tasks

## Learning Rate Decay

###  Why adjust learning rate?

- Early training → faster convergence
- Later training → slower convergence for stability

###  Step Learning Rate Decay
- lr = lr * gamma
- Uses fixed intervals to reduce learning rate.

```python
# Fixed interval: change the learning rate after a specified number of training steps
# lr = lr * gamma

import torch
from torch import optim
import matplotlib.pyplot as plt


def dm01():
	# todo: 1 - initialize parameters
	# lr, epoch, iteration
	lr = 0.1
	epoch = 200
	iteration = 10

	# todo: 2 - create dataset
	# y_true, x, w
	y_true = torch.tensor([0])
	x = torch.tensor([1.0], dtype=torch.float32)
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

	# todo: 3 - create optimizer object (momentum method)
	optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

	# todo: 4 - create fixed interval learning rate scheduler
	# optimizer: optimizer object
	# step_size: interval, change the learning rate after a specified number of training steps
	# gamma: decay factor, default is 0.1
	scheduer = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5)

	# todo: 5 - create two lists to collect training steps and learning rates
	lr_list, epoch_list = [], []

	# todo: 6 - iterate through training epochs
	for i in range(epoch):

		# todo: 7 - get the training step and learning rate and save them into lists
		# scheduer.get_last_lr(): get the latest learning rate
		lr_list.append(scheduer.get_last_lr())
		epoch_list.append(i)

		# todo: 8 - iterate through batches
		for batch in range(iteration):

			# first compute predicted y value wx, then compute loss (wx - y_true)^2
			y_pred = w * x
			loss = (y_pred - y_true) ** 2

			# clear gradients
			optimizer.zero_grad()

			# compute gradients
			loss.backward()

			# update parameters
			optimizer.step()

		# todo: 9 - update the learning rate for the next training step
		scheduer.step()
	print('lr_list->', lr_list)

	plt.plot(epoch_list, lr_list)
	plt.xlabel("Epoch")
	plt.ylabel("Learning rate")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	dm01()
```

### Multi-step Learning Rate Decay

- Learning rate changes at specific training steps.
- milestones=[50,100,160]

  ```python
  # Specified intervals: change the learning rate after certain training steps
  # using a step list  lr = lr * gamma

  import torch
  from torch import optim
  import matplotlib.pyplot as plt


  def dm01():
	# todo: 1 - initialize parameters
	# lr, epoch, iteration
	lr = 0.1
	epoch = 200
	iteration = 10

	# todo: 2 - create dataset
	# y_true, x, w
	y_true = torch.tensor([0])
	x = torch.tensor([1.0], dtype=torch.float32)
	w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

	# todo: 3 - create optimizer object (momentum method)
	optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

	# todo: 4 - create learning rate scheduler with specified intervals
	# optimizer: optimizer object
	# milestones: list of training steps where learning rate will change
	# gamma: decay factor, default is 0.1
	scheduer = optim.lr_scheduler.MultiStepLR(
		optimizer=optimizer,
		milestones=[50, 100, 160],
		gamma=0.5,
		last_epoch=-1
	)

	# todo: 5 - create two lists to store training steps and learning rates
	lr_list, epoch_list = [], []

	# todo: 6 - iterate through training epochs
	for i in range(epoch):

		# todo: 7 - record the training step and learning rate
		# scheduer.get_last_lr(): get the latest learning rate
		lr_list.append(scheduer.get_last_lr())
		epoch_list.append(i)

		# todo: 8 - iterate through batches
		for batch in range(iteration):

			# compute predicted value y (wx) and calculate loss (wx - y_true)^2
			y_pred = w * x
			loss = (y_pred - y_true) ** 2

			# clear gradients
			optimizer.zero_grad()

			# compute gradients
			loss.backward()

			# update parameters
			optimizer.step()

		# todo: 9 - update the learning rate for the next training iteration
  		scheduer.step()
  	print('lr_list->', lr_list)
  
  	plt.plot(epoch_list, lr_list)
  	plt.xlabel("Epoch")
  	plt.ylabel("Learning rate")
  	plt.legend()
  	plt.show()
  
  
  if __name__ == '__main__':
  	dm01()
  ```

### Exponential Learning Rate Decay

- lr = lr * gamma^epoch
- Early stage: fast decay
- Middle stage: slower
- Late stage: very slow

  ```python
  # Exponential decay: learning rate decreases quickly in the early stage,
  # slower in the middle stage, and even slower in the later stage
  # lr = lr * gamma**epoch

  import torch
  from torch import optim
  import matplotlib.pyplot as plt

  def dm01():

	# todo: 1 - initialize parameters
	# lr, epoch, iteration
	lr = 0.1
	epoch = 200
	iteration = 10

	# todo: 2 - create dataset
	# y_true, x, w
	y_true = torch.tensor([0])
	x = torch.tensor([1.0], dtype=torch.float32)
	w
  		scheduer.step()
  	print('lr_list->', lr_list)
  
  	plt.plot(epoch_list, lr_list)
  	plt.xlabel("Epoch")
  	plt.ylabel("Learning rate")
  	plt.legend()
  	plt.show()
  
  
  if __name__ == '__main__':
  	dm01()
  ```

###  Choosing Learning Rate Decay Methods

- Prefer Exponential decay
- Use Multi-step decay based on experience
- Use Step decay for simple models
