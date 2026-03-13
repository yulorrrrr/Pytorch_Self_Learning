"""
Example:
    Demonstration of the complete CNN workflow: image classification.

Overview: Steps of a deep learning project
    1. Prepare the dataset.
        Here we use the computer vision dataset CIFAR10 provided by torchvision. It contains 60,000 
        images of size (32, 32, 3), including: 50,000 training images and 10,000 test images.
        There are 10 classes, and each class contains about 6,000 images.
        You need to install the torchvision package first, e.g.:pip install torchvision

    2. Build a convolutional neural network.
    3. Train the model.
    4. Test the model.


Convolution Layer:
    Extract local features from the image → Feature Map
    Computation formula:
    N = (W - F + 2P) // S + 1
    Each convolution kernel corresponds to one neuron.


Pooling Layer:

    Dimensionality reduction, including:Max pooling and average pooling.
    Pooling only changes the height and width (H, W) of the feature map, but the number of channels does not change.

Optimization method:
    1. add the number of convolutional kernel
    2. add the number of fully connect layers
    3. change learning rate
    4. change optimizer
    5. change activation function
    6. ...

"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

#batch sample size
BATCH_SIZE = 8

#1. create dataset
def create_dataset():
    #1. get training set
    #param1: root: dataset storage path, param2: train: whether to load the training set
    #param3: download: whether to download the dataset, param4: transform: data transformation method
    train_data = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    #2. get test set
    test_data = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    #3. return dataset
    return train_data, test_data

#2. Build a convolutional neural network.
class ImageModel(nn.Module):
    # 1. initialize parent class and define layers
    def __init__(self):
        #1.1 initialize parent class
        super().__init__()
        #1.2 build neural network layers
        #convolutional layer 1 input3, output6, kernel size 3, stride 1, padding 0
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        #pooling layer 1, kernel size 2, stride 2, padding 0
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        #convolutional layer 2, input6, output16, kernel size 3, stride 1, padding 0
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        #pooling layer 2, kernel size 2, stride 2, padding 0
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        #fully connected layer （hidden layer) 1, input576, output120
        self.linear1 = nn.Linear(576, 120)
        #fully connected layer （hidden layer) 2, input120, output84
        self.linear2 = nn.Linear(120, 84)
        #fully connected layer (output layer), input84, output10
        self.output = nn.Linear(84, 10)
    
    #2. define forward function, compute the forward pass of the model
    def forward(self, x):
        # layer1: convolutional layer 1(weighted sum) + activation function (relu) + pooling layer 1
        #x = self.conv1(x) #convolutional layer 1 
        #x = torch.relu(x) #activation function (relu)
        #x = self.pool1(x) #pooling layer 1 ->
        x = self.pool1(torch.relu(self.conv1(x)))

        # layer2: convolutional layer 2(weighted sum) + activation function (relu) + pooling layer 2
        x = self.pool2(torch.relu(self.conv2(x)))

        #*** fully connected layer can only process 2d data, need to flatten the data before inputting to the fully connected layer
        #(8, 16, 6, 6) -> (8, 576)
        #param1: sample size (row), param2: feature size (column), -1: automatically calculation
        x = x.reshape(x.shape[0], -1) #8row, 576 column

        # layer3:  fully connected layer 1(weighted sum) + activation function (relu)
        x = torch.relu(self.linear1(x))

        # layer4:  fully connected layer 2(weighted sum) + activation function (relu)
        x = torch.relu(self.linear2(x))

        # layer5:  output layer(weighted sum) + activation function (softmax)
        #(only need weighted sum, since the loss function will apply softmax)
        return self.output(x)  #later use CrossEntropyLoss = softmax() + Loss computation


#3. Train the model.
def train(train_data):
    #1. create dataloader
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    #2. create model object
    model = ImageModel()
    #3. define loss function 
    criterion = nn.CrossEntropyLoss() #CrossEntropyLoss = softmax() + Loss conputation
    #4. define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #5. training loop
    #5.1 define number of epochs
    epochs = 10  #number of epochs
    #5.2 start training
    for epoch_idx in range(epochs):
        #5.2.1 define variabble, record total loss and batches for each epoch, training time
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        #5.2.2 Iterate through the data loader to get each batch of data
        for x, y in dataloader:
            #5.2.3 change to training mode
            model.train()
            #5.2.4 model prediction
            y_pred = model(x)
            #5.2.5 compute loss
            loss = criterion(y_pred, y)
            #5.2.6 zero gradient + backpropagation + update parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #5.2.7 count the number of correct predictions
            #print(y_pred)   #the prediction probabilities for each class of each image in the batch
           
            #argmax() return the index of the maximum value -> predict class
            #print(torch.argmax(y_pred, dim=-1)) #-1 means row

            #print truw value
            #print(y)
            #print(torch.argmax(y_pred, dim=-1)== y)
            #print((torch.argmax(y_pred, dim=-1)== y).sum())
            total_correct += (torch.argmax(y_pred,dim=-1)==y).sum()

            #5.2.8 count the total loss for this batch   (the avg loss of this batch * the number of samples in this batch)
            total_loss += loss.item() * len(y) #[the total loss for first batch] + [the total loss for second batch] + ... + [the total loss for last batch]

            #5.2.9 count the total number of samples for this batch
            total_samples += len(y)

            #break one batch for each epoach, inprove the training speed, only for testing
        
        #5.2.10 one epoach training is complete, print the training information for this epoch
        print(f'epoch: {epoch_idx+1}/{epochs}, loss: {total_loss/total_samples:.4f}, acc: {total_correct/total_samples: 2f} time: {time.time() - start:.2f} seconds, ')
        #break

    #6. Save model
    torch.save(model.state_dict(), './day07/model/image_model.pth')



#4. Test the model.
def evaluate(test_data):
    #1. create dataloader
    dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    #2. create model object
    model = ImageModel()
    #3. upload model parameter
    model.load_state_dict(torch.load('./day07/model/image_model.pth'))  #pickle
    #4. define total corrct and total sample
    total_correct, total_sample = 0,0
    #5. Iterate through the data loader to get each batch of data
    for x, y in dataloader:
        #5.1 change to test mode
        model.eval()
        #5.2 predict model
        y_pred = model(x)
        #5.3 because CrossEntropyLoss is used in training, softmax() is already include in it
        #argmax(): return the index of maximum value
        y_pred = torch.argmax(y_pred, dim=-1) #-1: row
        #5.4 count the numer of correct prediction
        total_correct += (y_pred == y).sum()
        #5.5 count total sample
        total_sample += len(y)
    
    #6. print accuracy
    print(f'accuracy = {total_correct/total_sample:.2f}')


#5. test
if __name__ == '__main__':
    #1. create dataset
    train_data, test_data = create_dataset()
    # print(f'train_data: {train_data.data.shape}, test_data: {test_data.data.shape}')
    # print(f'type of dataset: {train_data.class_to_idx}')

    # #show img
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_data.data[0])
    # plt.title(train_data.targets[0])
    # plt.show()

    #2. create neural network model object
    #model = ImageModel()
    #check model parameters
    #param1: model param2: input dimension(CHW) param3: batch size
    #summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)

    #3. model training
    #train(train_data)

    #4. model testing
    evaluate(test_data)





