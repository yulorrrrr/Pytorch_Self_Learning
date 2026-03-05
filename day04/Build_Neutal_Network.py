'''
process: 1. Define a class that inherits from nn.Module
         2. Build neutal network in __init()__
         3. Complete forward pass in forwrd()
'''

import torch
import torch.nn as nn
from torchsummary import summary

# todo1. Build neutal network: inherit nn.Module
class ModelDemo(nn.Module):
    #1 Complete initialization in __init__: initialize the parent class and build the neural network
    def __init__(self):
        #1.1 initialize parent class
        super().__init__()
        #1.2 build neural network -> hidden layer + output layer
        #hidden layer1: input 3, output 3
        self.linear1 = nn.Linear(3,3)
        #hidden layer2: input 3, output 2
        self.linear2 = nn.Linear(3,2)
        #output layer: input 2, output 2
        self.output = nn.Linear(2,2)

        #1.3 Initialize hidden layer
        #Hidden layer 1:
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        #Hidden layer 2:
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    #todo 1.2: forward pass: input layer -> hidden layer -> output layer
    def forward(self, x):
        #1.1 First Layer: Hidden layer calculation: weighted sum + activation function(sigmoid)
        #Decomposed version
        #x = self.linear1(x) #weighted sun
        #x = torch.sigmoid(x) #activation function

        #Combine version
        x = torch.sigmoid(self.linear1(x))

        #1.2 Second Layer: Hidden layer calculation: weighted sum + activation function(ReLU)
        x = torch.relu(self.linear2(x))

        #1.3 Third Layer: Output layer: weighted sum + activation function(Softmax)
        #dim = -1, Indicates computation by rows, processing one sample at a time.
        x = torch.softmax(self.output(x), dim = -1)

        #1.4 return predict value
        return x

#todo 2: training model
def train():
    #1. create model
    my_model = ModelDemo()
    
    #2. create database sample, random generate
    data = torch.randn(size = (5,3)) 
    print(f'data: {data}')
    print(f'data.shape: {data.shape}')
    print(f'data.requires_grad: {data.requires_grad}')

    #3. use neural network model for model training
    output = my_model(data) #The forward() method is automatically called internally to perform the forward propagation.
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')
    print(f'output.requires_grad: {output.requires_grad}')
    print('-' * 30)

    #Compute and check model parameters
    print('=' * 20 + 'compute model parameter' +'=' *20)
    #param1:(neural network)model object, param2: input data dimension (5 row 3 column)
    summary(my_model, input_size=(5,3))

    print('=' * 20 + 'check model parameter' +'=' *20)
    for name, param in my_model.named_parameters():
        print(f'name: {name} \nparam: {param}\n')

#train
if __name__ == '__main__':
    train()