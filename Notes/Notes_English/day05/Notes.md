# Notes

## Regularization Methods

### What is Regularization

- Prevent model overfitting (good performance on the training set but poor performance on the test set), and improve the model’s generalization ability.
- A strategy used to prevent overfitting and improve generalization.
  - L1 regularization: needs to be implemented manually in code
  - L2 regularization: SGD(weight_decay=...)
  - Dropout
  - BN

###  Dropout Regularization

- Randomly deactivate neurons with probability p. During each batch of training, the neurons that are deactivated are random, which prevents the prediction result from depending too much on certain neurons (thus preventing overfitting).
- Probability p is usually in [0.2, 0.5]; simple models use a lower probability, while complex models use a higher one.
- The outputs of neurons that are not dropped are divided by (1-p) so that the output during training matches the output during testing (when dropout is inactive).
  - Training mode -> model.train()
  - Testing mode -> model.eval()
- Dropout is applied after the activation layer.

  ```python
  import torch
  import torch.nn as nn

  # Dropout randomly deactivates neurons: during batch training, it randomly
  # kills some neurons to prevent certain features from having too much influence
  # on the result (thus preventing overfitting)
  def dm01():
	# todo: 1 - create the hidden layer output
	# float(): convert to a floating-point tensor
	t1 = torch.randint(low=0, high=10, size=(1, 4)).float()
	print('t1->', t1)
	# todo: 2 - perform the weighted sum computation for the next layer
	linear1 = nn.Linear(in_features=4, out_features=4)
	l1 = linear1(t1)
	print('l1->', l1)
	# todo: 3 - compute the activation values
	output = torch.sigmoid(l1)
	print('output->', output)
	# todo: 4 - apply dropout to the activation values during training
	# p: dropout probability
	dropout = nn.Dropout(p=0.4)
	d1 = dropout(output)
	print('d1->', d1)


  if __name__ == '__main__':
  	dm01()
  ```

### Batch Normalization Regularization (Batch Normalization)

- Compute the mean and standard deviation of each batch of samples, and use them to calculate normalized values.

- Since each batch has different means and standard deviations, this introduces noisy sample data and reduces the training effect of the model (thus preventing overfitting).

- Introduce two learnable parameters, γ and β, so that the sample distribution of each layer can be different (the activation functions of each layer can also be different).

- Accelerate model training. The more uniform the data distribution is, the more likely the weighted sum results fall into a reasonable range (where the derivative is largest).

- Normalization is performed during training, but not during testing.

  ```python
  """
  Regularization: the mean and variance of each batch are different, which introduces noisy samples.
  Speeding up model convergence: after sample normalization, the values fall into a reasonable range of the activation function, where the derivative is as large as possible.
  """
  import torch
  import torch.nn as nn

  # nn.BatchNorm1d(): processes 1D samples; each batch must contain at least
  # 2 samples, otherwise the mean and standard deviation cannot be calculated
  # nn.BatchNorm2d(): processes 2D samples, such as images
  # (each channel is a 2D matrix), and computes the mean and standard deviation
  # of each column in the 2D matrix
  # nn.BatchNorm3d(): processes 3D samples, such as videos
  # Processing 2D data
  def dm01():
	# todo: 1 - create an image sample dataset: 2 channels, each channel has
	# a 3*4 feature map, representing a feature map sample processed by a conv layer
	# The dataset contains only one image, composed of 2 channels,
	# and each channel is a 3*4 pixel matrix
	input_2d = torch.randn(size=(1, 2, 3, 4))
	print('input_2d->', input_2d)
	# todo: 2 - create a BN layer; normalization must be done before the activation function
	# num_features: number of channels in the input sample
	# eps: a small constant to avoid division by zero
	# momentum: exponential moving weighted average
	# affine: True by default, introduces learnable γ and β parameters
	bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)
	ouput_2d = bn2d(input_2d)
	print('ouput_2d->', ouput_2d)

  # Processing 1D data
  def dm02():
	# Create the sample dataset
	input_1d = torch.randn(size=(2, 2))
	# Create the linear layer
	linear1 = nn.Linear(in_features=2, out_features=4)
	l1 = linear1(input_1d)
	print('l1->', l1)
	# Create the BN layer
	bn1d = nn.BatchNorm1d(num_features=4)
	# Normalize the output of the linear layer
	output_1d = bn1d(l1)
	print('output_1d->', output_1d)

  if __name__ == '__main__':
	# dm01()
	dm02()
  ```

## Mobile Phone Price Classification Case
### Case Requirements

- A classification problem with four classes: 0, 1, 2, and 3
- Implementation steps
  - Prepare the dataset -> split the dataset and convert it into a tensor dataset
  - Build the neural network model -> inherit from nn.Module
  - Train the model
  - Evaluate the model

### Build the Tensor Dataset

```python
# Import related modules
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time


# todo: 1 - build the dataset
def create_dataset():
    print('===========================Build Tensor Dataset Object===========================')
	# todo: 1-1 load the CSV dataset file
	data = pd.read_csv('data/手机价格预测.csv')
	print('data.head()->', data.head())
	print('data.shape->', data.shape)
	# todo: 1-2 get the feature dataset x and target dataset y
	# iloc: index-based selection
	x, y = data.iloc[:, :-1], data.iloc[:, -1]
	# convert feature columns to float type
	x = x.astype(np.float32)
	print('x->', x.head())
	print('y->', y.head())
	# todo: 1-3 split the dataset 8:2
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
	# todo: 1-4 convert the dataset to a tensor dataset
	# x_train and y_train are DataFrame objects, which cannot be directly converted to tensors
	# x_train.values(): get the data values from the DataFrame, returning a NumPy array
	# torch.tensor(): convert a NumPy array to a tensor
	train_dataset = TensorDataset(torch.tensor(data=x_train.values), torch.tensor(data=y_train.values))
	valid_dataset = TensorDataset(torch.tensor(data=x_valid.values), torch.tensor(data=y_valid.values))
	# todo: 1-5 return the training dataset, test dataset, number of features, and number of classes
	# shape -> (rows, columns), [1] accesses the tuple index
	# np.unique() -> remove duplicates; len() -> number of unique classes
	print('x.shape[1]->', x.shape[1])
	print('len(np.unique(y)->', len(np.unique(y)))
	return train_dataset, valid_dataset, x.shape[1], len(np.unique(y))


if __name__ == '__main__':
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
```

### Build the Classification Neural Network Model

```python
# todo: 2 - build the neural network classification model
class PhonePriceModel(nn.Module):
	print('===========================Build Neural Network Classification Model===========================')
	# todo: 2-1 build the neural network in __init__()
	def __init__(self, input_dim, output_dim):
		# inherit the constructor of the parent class
		super().__init__()
		# first hidden layer
		self.linear1 = nn.Linear(in_features=input_dim, out_features=128)
		# second hidden layer
		self.linear2 = nn.Linear(in_features=128, out_features=256)
		# output layer
		self.output = nn.Linear(in_features=256, out_features=output_dim)
	# todo: 2-2 forward propagation method forward()
	def forward(self, x):
		# computation of the first hidden layer
		x = torch.relu(input=self.linear1(x))
		# computation of the second hidden layer
		x = torch.relu(input=self.linear2(x))
		# computation of the output layer
		# Softmax is not applied here because CrossEntropyLoss later combines
		# softmax and loss computation
		output = self.output(x)
		return output
# todo: 3 - model training
# todo: 4 - model evaluation

if __name__ == '__main__':
	# Create the tensor dataset object
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# Create the model object
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# Calculate model parameters
	# input_size: shape of the input sample
	summary(model, input_size=(16, input_dim))
```

###  Model Training

```python
# todo: 3 - model training
def train(train_dataset, input_dim, class_num):
	print('===========================Model Training===========================')
	# todo: 3-1 create a data loader for batch training
	dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
	# todo: 3-2 create the neural network classification model object and initialize w and b
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	print("======View model parameters w and b======")
	for name, parameter in model.named_parameters():
		print(name, parameter)
	# todo: 3-3 create the loss function object
	# multi-class cross-entropy loss = softmax + loss computation
	criterion = nn.CrossEntropyLoss()
	# todo: 3-4 create the optimizer object SGD
	optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
	# todo: 3-5 model training using mini-batch stochastic gradient descent
	# number of training epochs
	num_epoch = 50
	for epoch in range(num_epoch):
		# define variables to accumulate total loss and number of batches
		total_loss = 0.0
		batch_num = 0
		# training start time
		start = time.time()
		# batch training
		for x, y in dataloader:
			# switch the model to training mode
			model.train()
			# model prediction
			y_pred = model(x)
			# print('y_pred->', y_pred)
			# compute loss
			loss = criterion(y_pred, y)
			# print('loss->', loss)
			# zero the gradients
			optimizer.zero_grad()
			# compute gradients
			loss.backward()
			# update parameters using gradient descent
			optimizer.step()
			# accumulate the average loss and batch count
			# item(): get the value of a scalar tensor
			total_loss += loss.item()
			batch_num += 1
		# print loss changes
		print('epoch: %4s loss: %.2f, time: %.2fs' % (epoch + 1, total_loss / batch_num, time.time() - start))
	# todo: 3-6 save the model by storing the parameter dictionary and then saving it to a file
	torch.save(model.state_dict(), 'model/phone.pth')


if __name__ == '__main__':
	# Create the tensor dataset object
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# Create the model object
	# model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# Calculate model parameters
	# input_size: shape of the input sample
	# summary(model, input_size=(16, input_dim))
	# Model training
	train(train_dataset=train_dataset, input_dim=input_dim, class_num=class_num)
```

### Model Evaluation

```python
# todo: 4 - model evaluation
def test(valid_dataset, input_dim, class_num):
	# todo: 4-1 create the neural network classification model object
	model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# todo: 4-2 load the trained model parameter dictionary
	model.load_state_dict(torch.load(f='model/phone.pth'))
	# todo: 4-3 create a data loader for the test set
	# shuffle does not need to be True for prediction
	dataloader = DataLoader(dataset=valid_dataset, batch_size=8, shuffle=False)
	# todo: 4-4 define a variable initialized to 0 to count correctly predicted samples
	correct = 0
	# todo: 4-5 predict batch by batch
	for x, y in dataloader:
		print('y->', y)
		# switch the model to evaluation mode
		model.eval()
		# model prediction -> weighted sum at the output layer
		output = model(x)
		print('output->', output)
		# get the predicted class from the weighted sums
		# argmax() returns the index of the maximum value, which is the class
		# y -> 0, 1, 2, 3
		# dim=1: process row by row, one sample at a time
		y_pred = torch.argmax(input=output, dim=1)
		print('y_pred->', y_pred)
		# count the number of correctly predicted samples
		print(y_pred == y)
		# sum boolean values: True -> 1, False -> 0
		print((y_pred == y).sum())
		correct += (y_pred == y).sum()
		print('correct->', correct)
	# calculate prediction accuracy
	print('Acc: %.5f' % (correct.item() / len(valid_dataset)))


if __name__ == '__main__':
	# Create the tensor dataset object
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	# Create the model object
	# model = PhonePriceModel(input_dim=input_dim, output_dim=class_num)
	# Calculate model parameters
	# input_size: shape of the input sample
	# summary(model, input_size=(16, input_dim))
	# Model training
	# train(train_dataset=train_dataset, input_dim=input_dim, class_num=class_num)
	# Model evaluation
	test(valid_dataset=valid_dataset, input_dim=input_dim, class_num=class_num)
```

### Network Performance Optimization

- Standardize the input layer data
- Increase the number of neural network layers and the number of neurons
- Change the gradient descent optimization method from SGD to Adam
- Change the learning rate from 1e-3 to 1e-4
- Regularization
- Increase the number of training epochs

...

## Basic Image Knowledge

### 3.1 Image Concepts

- Image categories in computers
  - Binary image: 1 channel (1 two-dimensional matrix), pixel values: 0 or 1
  - Grayscale image: 1 channel, pixel values: 0–255
  - Indexed image: 1 channel, index value -> row index of the RGB 2D matrix, color image pixel values: 0–255
  - RGB true-color image (most commonly used): 3 channels (3 two-dimensional matrices), R, G, and B matrices, pixel values: 0–255

###  Image Loading

```python
import numpy as np
import matplotlib.pyplot as plt
import torch


# Create a fully black and a fully white image
def dm01():
	# Fully black image
	# Create a 3-channel 2D matrix, black means pixel value 0
	# H W C: 200, 200, 3
	# Height, Width, Channel
	img1 = np.zeros(shape=(200, 200, 3))
	print('img1->', img1)
	print('img1.shape->', img1.shape)
	# Display image
	plt.imshow(img1)
	plt.show()

	# Fully white image
    # If fed into a fully connected layer, it becomes a 1D vector of 200*200*3=120000 features
	img2 = torch.full(size=(200, 200, 3), fill_value=255)
	print('img2->', img2)
	print('img2.shape->', img2.shape)
	# Display image
	plt.imshow(img2)
	plt.show()


def dm02():
	# Load image
	img1 = plt.imread(fname='data/img.jpg')
	print('img1->', img1)
	print('img1.shape->', img1.shape)
	# Save image
	plt.imsave(fname='data/img1.png', arr=img1)
	# Display image
	plt.imshow(img1)
	plt.show()


if __name__ == '__main__':
	# dm01()
	dm02()
```

## Introduction to Convolutional Neural Networks (CNN)

### What is CNN

- A neural network model that includes convolutional layers, pooling layers, and fully connected layers
- Components
  - Convolutional layer: extracts image feature maps
  - Pooling layer: reduces dimensionality, decreases the number of feature values in the feature map, and reduces model parameters
  - Fully connected layer: performs prediction and can only accept a two-dimensional dataset, where one sample is a one-dimensional vector
    - Convert the pooled feature map of one image into a one-dimensional vector: 200*200*3 -> 120000 feature values

### CNN Application Scenarios

- Image classification
- Object detection
- Face unlocking
- Autonomous driving
- ...

## Convolutional Layer

> Function: extract feature maps

### Convolution Computation

> Convolution computation is equivalent to the weighted-sum computation in a linear layer

- Perform dot-product operations between a weighted convolution kernel and the feature values of the image to obtain one feature value in the new feature map

- Convolution kernel / filter -> a neuron with weight parameters

- w1x1 + w2x2 + .... + b

  - w1-> one weight parameter of the convolution kernel
  - x1-> one feature value (pixel) in the feature map

  ![1737367789548](assets/1737367789548.png)

### Padding

- Add feature values around the original image feature map (default is padding with 0)
- Functions
  - Keep the shape of the new feature map consistent with the original feature map
  - Reduce the loss of edge feature information
    - Without padding, edge feature values participate in the computation only once; after padding, edge feature values participate multiple times
- Implementation methods
  - No padding: the new feature map is smaller than the original image feature map
  - Same padding: the original and new feature maps have the same shape
  - Full padding: the new feature map is larger than the original image feature map, adding new features

### Stride

- Stride refers to the step size of the convolution kernel (neuron) as it slides over the feature map; the default value is 1
- Functions:
  - Reduce computation
  - Reduce features, meaning the new feature map has fewer feature values (dimensionality reduction)
- Usually the default is 1, but it can be set to 2 or 4
- For an original feature map of 5*5:
  - stride=1 -> new feature map 3*3
  - stride=2 -> new feature map 2*2

### Multi-Channel Convolution Computation

- An RGB color image consists of 3 channels -> 3 two-dimensional matrices, representing R/G/B respectively
- The number of channels in the convolution kernel is the same as the number of channels in the original image
- Convolution computation means performing convolution on the corresponding 2D matrix of each channel, then adding the results of all channels together to obtain one feature value in the new feature map
- The new feature map is one 2D matrix, not three 2D matrices

### Multi-Kernel Convolution Computation

- The number of convolution kernels equals the number of neurons, and determines how many two-dimensional feature maps will be extracted

### Feature Map Size Calculation

- N = (W-F+2P)/S + 1
- N: height or width of the new feature map
- W: height or width of the original feature map
- F: height or width of the convolution kernel
- P: padding value
- S: stride value
- If N is a decimal, round it down using the built-in floor function

### Convolution Layer API Usage

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
"""
in_channels: number of channels in the original image; RGB color images have 3
out_channels: number of convolution kernels / neurons; the output image consists of n-channel 2D matrices
kernel_size: shape of the convolution kernel, e.g. (3,3), (3,5)
stride: step size, default is 1
padding: number of padding layers, default is 0; 1; same -> stride=1; 2, 3...
nn.Conv2d(in_channels=, out_channels=, kernel_size=, stride=, padding=)
"""


def dm01():
	# todo: 1 - load an RGB color image (H, W, C)
	img = plt.imread(fname='data/img.jpg')
	print('img->', img)
	print('img.shape->', img.shape)
	# todo: 2 - convert the image shape from (H, W, C) to (C, H, W) using permute()
	img2 = torch.tensor(data=img, dtype=torch.float32).permute(dims=(2, 0, 1))
	print('img2->', img2)
	print('img2.shape->', img2.shape)
	# todo: 3 - save this image into the dataset as (batch_size, C, H, W) using unsqueeze()
	# The dataset contains only one sample
	img3 = img2.unsqueeze(dim=0)
	print('img3->', img3)
	print('img3.shape->', img3.shape)
	# todo: 4 - create the convolution layer object to extract feature maps
	conv = nn.Conv2d(in_channels=3,
					 out_channels=4,
					 kernel_size=(3, 3),
					 stride=2,
					 padding=0)
	conv_img = conv(img3)
	print('conv_img->', conv_img)
	print('conv_img.shape->', conv_img.shape)

	# View the 4 extracted feature maps
	# Get the first image in the dataset
	img4 = conv_img[0]
	# Convert shape to (H, W, C)
	img5 = img4.permute(1, 2, 0)
	print('img5->', img5)
	print('img5.shape->', img5.shape)
	# img5 -> (H, W, C)
	# img5[:, :, 0] -> the 2D matrix feature map of the first channel, i.e. the first feature map
	feature1 = img5[:, :, 0].detach().numpy()
	plt.imshow(feature1)
	plt.show()


if __name__ == '__main__':
	dm01()
```

## Pooling Layer

> The pooling layer does not involve neurons. It is only used for dimensionality reduction and does not perform feature extraction.

### Pooling Computation

- Perform dimensionality reduction on the feature maps extracted by the convolution layer
- Max pooling -> uses the largest feature value in the 2D matrix as the output feature
- Average pooling -> uses the average feature value in the 2D matrix as the output feature

###  Multi-Channel Pooling Computation

- Pooling is performed only along the height and width dimensions; the channel dimension does not participate in pooling
- No matter how many channels the feature maps extracted by the convolution layer have, the number of channels remains the same after pooling

### Pooling Layer API Usage

```python
import torch
import torch.nn as nn
"""
Max pooling
kernel_size: the size of the window, not the size of a neuron, because pooling layers do not involve neurons
nn.MaxPool2d(kernel_size=, stride=, padding=)

Average pooling
nn.AvgPool2d(kernel_size=, stride=, padding=)
"""

# Pooling on a single-channel convolutional feature map
def dm01():
	# Create a 1-channel 3*3 2D matrix, i.e. one feature map
	inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=torch.float)
	print('inputs->', inputs)
	print('inputs.shape->', inputs.shape)
	# Create the pooling layer
	# kernel_size: the size of the window
	pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool1(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)
	pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool2(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)


# Pooling on multi-channel convolutional feature maps
def dm02():
	# size(3,3,3)
	inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
						   [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
						   [[11, 22, 33], [44, 55, 66], [77, 88, 99]]], dtype=torch.float)
	# Create the pooling layer
	# kernel_size: the size of the window
	pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool1(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)
	pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=0)
	outputs = pool2(inputs)
	print('outputs->', outputs)
	print('outputs.shape->', outputs.shape)


if __name__ == '__main__':
	# dm01()
	dm02()
```

