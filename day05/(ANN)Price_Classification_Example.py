"""
Example:
ANN (Artificial Neural Network) case: Mobile phone price classification.

Background:
Based on 20 features of mobile phones, predict the price range of the phone
(4 price categories). This task can be solved using machine learning
or deep learning (recommended).
"""
import torch
from torchsummary import summary #model architecture visualization
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

#todo 1: Define function, build dataset
def create_dataset():
    #1. Load CSV dataset
    data = pd.read_csv('./day05/data/mobile_price_classification.csv')
    #print(f'data: {data.head()}')
    #print(f'data shape: {data.shape}') #(2000, 21) -> 2000 samples, 20 features + 1 label

    #2. get x characteristics and y label
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    #print(f'x: {x.head()}, {x.shape()}')
    #print(f'y: {y.head()}, {y.shape()}')

    #3.turn characteristics row to float
    x = x.astype(np.float32)

    #4. split dataset into training set and test set
    #param1: characteristics, param2: label, param3: test set size, param4: random state for reproducibility, param5: stratify by label to maintain class distribution
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    #5. convert data to tensor and create dataset and dataloader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    #print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    #6. return                    20: serve as the number of input features, 4: serve as the number of input features
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

#todo 2： Define function, build ANN model
class PhonePriceModel(nn.Module):
    #1.  initialize parent class and define layers
    def __init__(self, input_dim, output_dim): #input: 20, output:4
        #1.1 initialize parent class
        super().__init__()
        #1.2 build neural network layers
        #hidden layer 1:
        self.linear1 = nn.Linear(input_dim, 128)
        #hidden layer 2:
        self.linear2 = nn.Linear(128, 256)
        #output layer:
        self.output = nn.Linear(256, output_dim)   
    
    #2. define forward function, compute the forward pass of the model
    def forward(self, x):
        #2.1 hidden layer 1: weighted sum + activation function(relu)
        x = torch.relu(self.linear1(x))

        #2.2 hidden layer 2: weighted sum + activation function(relu)
        x = torch.relu(self.linear2(x))

        #2.3 output layer: weighted sum + activation function(softmax) 
        #(only need weighted sum, since the loss function will apply softmax)
        x = self.output(x)

        #2.4 return output
        return x
    
#todo 3: Define function, train the model
def train(train_dataset, input_dim, output_dim):
    #1. create dataloader data -> tensor -> dataset -> dataloader
    #param1: dataset(1600), param2: batch size, param3: whether to shuffle the data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #2. create model object
    model = PhonePriceModel(input_dim, output_dim)
    #3. define loss function and optimizer 
    criterion = nn.CrossEntropyLoss() #multi-class classification (alreafy includes softmax)
    optimizer = optim.SGD(model.parameters(), lr=0.001) #learning rate = 0.01
    #4. training loop
    #4.1 define number of epochs
    epochs = 50
    #4.2 start training
    for epoch in range(epochs):
        #4.2.1 define variable to record total loss and batches for each epoch
        total_loss, batch_num = 0, 0
        #4.2.2 define variable to record start time for each epoch
        start_time = time.time()
        #4.2.3 start training for each batch
        for x, y in train_loader:
            #4.2.4 change model mode (training and testing)
            model.train() #training mode    model.eval() #testing mode
            #4.2.5 compute prediction and loss
            y_pred = model(x)
            #5.2.6 compute loss
            loss = criterion(y_pred, y)
            #4.2.7 zero gradient + backpropagation + update parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #4.2.8 update total loss and batch count
            total_loss += loss.item() #accumulate the average loss of each batch (16 samples) in this epoch
            batch_num += 1
        #4.2.4 The training for this epoch is completed.
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {total_loss/batch_num:.4f}, Time: {time.time() - start_time:.2f} seconds')

    #5. After multiple training epochs, save the model.
    #param1: model parameters, param2: file path to save the model
    print(f'\n\n the parameters in the model: {model.state_dict()}\n\n')
    torch.save(model.state_dict(), './day05/model/phone_price_model.pth')

#todo 4: Define function, test the model
def evaluate(test_dataset, input_dim, output_dim):
    #1. create neural network model object
    model = PhonePriceModel(input_dim, output_dim)
    #2. load the trained model parameters
    model.load_state_dict(torch.load('./day05/model/phone_price_model.pth'))
    #3. create dataloader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) #train: suffle = True, test: shuffle = False
    #4. define variable to record correct predictions and total samples
    correct = 0
    #5. store data for each batch from dataloader
    for x, y in test_loader:
        #5.1 change model mode to testing
        model.eval()
        #5.2 predict model
        y_pred = model(x)
        #5.3 Determine the class based on the weighted sum, and use argmax() to obtain the index 
        #of the maximum value, which represents the predicted class.
        y_pred = torch.argmax(y_pred, dim=1) #dim = -1 means processing row by row.
        print(f'y: {y}') #the true class of the first sample / second sample / ... in this batch           
        print(f'y_pred: {y_pred}') #the predicted class of the first sample / second sample / ... in this batch
        #5.4 count the number of correctly predicted samples
        #print(y_pred == y)
        #print((y_pred == y).sum())
        correct += (y_pred == y).sum()

    #model predict ends, print accuracy
    print(f'accuracy: {correct/len(test_dataset):.4f}')

#todo 5: test
if __name__ == '__main__':
    #1. create dataset
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    #print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')
    #print(f'input_dim: {input_dim}, output_dim: {output_dim}')

    #2. build neural network model
    #model = PhonePriceModel(input_dim, output_dim)
    # compute the parameters in the model
    #summary(model, input_size = (16, input_dim)) #input_size: (batch_size, input characteristics)
    
    #3. train the model
    #train(train_dataset, input_dim, output_dim)

    #4. test the model
    evaluate(test_dataset, input_dim, output_dim)