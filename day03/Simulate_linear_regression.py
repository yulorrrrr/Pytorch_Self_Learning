'''
The model contruction process on pytorch:
1. prepare dataset
2. design model using class
3. Construct loss and optimizer
4. traning cycle

**Mean Squared Error (MSE) Loss: criterion = nn.MSELoss()
**optimizer: optimizer = optim.SGD(params=model.parameters(), 1r=1e-2)
'''

import torch
import matplotlib.pyplot as plt #Visualization
from torch.utils.data import TensorDataset #Construct the dataset
from torch.utils.data import DataLoader #dataloader
from torch import nn #The nn module provides the squared loss function and the hypothesis (model) function.
from torch import  optim #The optim module contains optimization algorithms (optimizers).  
from sklearn.datasets import make_regression #the nn module contains the squared loss function and the hypothesis function

#numpy -> Tensor -> TensorDataset -> DataLoader

#1. define function, create linear regression sample
def create_dataset():
    x, y, coef = make_regression(
        n_samples=100, #100 sample
        n_features=1,  #1 feature
        noise=10, #The larger the noise, the more scattered the data points; #the smaller the noise, the more concentrated the data points.
        coef=True, #Wheather to return t he coefficient, default: false
        bias=14.5,  #bias
        random_state=3 #random seed
    )

    #convert the above data to tensor
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x,y, coef

#2. Define function for model training
def train(x,y,coef):
    #1. create database object, convert tensor -> database object -> dataloader
    dataset = TensorDataset(x,y)
    
    #2. create dataloader
    #shuffle: training set: True, Test set: False
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    #3. create initial linear regression model
    #param1: input dim, param2: output dim
    model = nn.Linear(1,1)

    #4. create loss object
    criterion = nn.MSELoss()

    #5. create optimizer
    #param1: model parameters, param2: learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #6. detailed training process
    #6.1: define variables: training times, the loss for each train(avg), total loss, number of samples
    epoch, loss_list, total_loss, total_sample = 100, [], 0, 0

    #6.2: start training:
    for epoch in range(epoch):
        #6.3 Each epoch is trained in batches, get batches data from dataloader
        for train_x, train_y in dataloader: #7 batches(16*6 + 4)

            #6.4 predict model
            y_predict = model(train_x)

            #6.5 calculate (avg) loss
            loss = criterion(y_predict, train_y.reshape(-1, 1)) #n row 1 column
            
            #6.6 compute total loss and sample(batches) size
            total_loss += loss.item()
            total_sample += 1

            #6.7 reset grad + backward + update grad
            optimizer.zero_grad() #reset grad
            loss.backward()       #backward, calculate grad
            optimizer.step()      #update grad

        #6.8 add (avg) loss to the list
        loss_list.append(total_loss / total_sample)
        print(f'round: {epoch + 1}, avg loss: {total_loss / total_sample}')
        
    #print training result：
    print(f'{epoch} round of average loss is: {loss_list}')
    print(f'model parameter, weight: {model.weight}, bias: {model.bias}')

if __name__ == '__main__':
    x, y, coef = create_dataset()
    #print(f'x: {x}, y: {y}, coef: {coef}')
    
    #model training
    train(x,y,coef)