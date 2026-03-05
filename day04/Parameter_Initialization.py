'''
Purpose for parameter initialization:
    1. Prevent vanishing gradients or exploding gradients
    2. Improve convergence speed
    3. Break Symmetry

Parameter Initialization method:
    1. Can't break symmetry:
        all 0 / all 1 : very simple
    2. Can break symmetry:
        Random Initialization /  Normal Initialization / kaiming Initialization /  xavier Initialization

How to choose:
    1. ReLU -> kaiming
    2. not ReLU -> xavier
    3. Shallow Neural Network (<5)-> Random Initialization
'''

import torch.nn as nn  #neural network

#1.Uniform Random Initialization
def dm01():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), generating parameters from a uniform distribution between 0 and 1.
    nn.init.uniform_(linear.weight)
    #3. Initialize bias(b), generating parameters from a uniform distribution between 0 and 1.
    nn.init.uniform_(linear.bias)
    #print
    print(linear.weight.data)
    print(linear.bias.data)


#2. Constant Initialization
def dm02():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), constant: 3.
    nn.init.constant_(linear.weight, 3)
    #3. Initialize bias(b), constant: 3.
    nn.init.constant_(linear.bias, 3)
    #print
    print(linear.weight.data)
    print(linear.bias.data)

#3. Zero Initialization
def dm03():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), zero initialization.
    nn.init.zeros_(linear.weight)
    #3. Initialize bias(b), zero initialization.
    nn.init.zeros_(linear.bias)
    #print
    print(linear.weight.data)
    print(linear.bias.data)

#4. One Initialization
def dm04():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), one initialization.
    nn.init.ones_(linear.weight)
    #print
    print(linear.weight.data)

#5. Normal Initialization
def dm05():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), normal initialization(default: mean:0, standard deviation:1).
    nn.init.normal_(linear.weight)
    #print
    print(linear.weight.data)

#6. kaiming Initialization
def dm06():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), normal initialization(default: mean:0, standard deviation:1).
    #kaiming normal initialization
    nn.init.kaiming_normal(linear.weight)
    #kaiming uniform initialization
    #nn.init.kaiming_uniform_(linear.weight)
    #print
    print(linear.weight.data)

#7. xavier Initialization
def dm07():
    #1. Create a linear layer, input feature: 5, output feature: 3
    linear = nn.Linear(5,3)
    #2. Initialize weight(w), normal initialization(default: mean:0, standard deviation:1).
    #xavier normal initialization
    nn.init.xavier_normal_(linear.weight)
    #xavier uniform initialization
    #nn.init.xavier_uniform_(linear.weight)
    #print
    print(linear.weight.data)

#test
if __name__ == '__main__':
    #dm01()  #Uniform Random Initialization
    #dm02() #Constant Initialization
    #dm03() #Zero Initialization
    #dm04() #One Initialization
    #dm05() #Normal Initialization
    #dm06() #kaiming Initialization
    dm07() #xavier Initialization
    