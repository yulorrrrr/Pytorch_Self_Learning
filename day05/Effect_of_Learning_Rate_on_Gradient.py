'''
Demonstrate learning rate decay strategies.

Introduction to learning rate decay strategies:
    Objective:
        Compared with AdaGrad, RMSProp, and Adam, we can manually control the adjustment of the learning rate using:
            step interval decay
            specified interval decay
            exponential decay
    
    Categories:
        Step interval learning rate decay
        Specified interval learning rate decay
        Exponential learning rate decay
    
Step interval learning rate decay:
    step_size: number of intervals (how many epochs before adjusting the learning rate)
    gamma: learning rate decay coefficient: lr_new = lr_old * gammma

Specified interval learning rate decay:
    milestones = [50, 100, 150] : a list of epoch indices. When the current epoch index matches any 
                 value in milestones, the learning rate is adjusted according to the decay coefficient gamma.
    gamma: learning rate decay coefficient: lr_new = lr_old * gammma

Exponential learning rate decay:
    Fast decay in the early stage, slower in the middle stage, and even slower in the later stage.
    Formula:
    lr_new = lr_old * gamma ** epoch

Summary:
    Step Learning Rate Decay (Fixed Interval Decay)
        Advantages:
            Intuitive and easy to debug
            Suitable for large datasets
        Disadvantages:
            The learning rate changes significantly, which may cause the optimizer to skip the 
            optimal solution
        Application Scenarios:
            Large datasets
            Relatively simple tasks
            
    Specified Interval Learning Rate Decay
        Advantages:
            Easy to debug
            Helps stabilize the training process
        Disadvantages:
            In some cases the decay may occur too quickly, causing the optimization to stop improving prematurely
        Application Scenarios:
            Tasks that require stable training
    
    Exponential Learning Rate Decay
        Advantages:
            Smooth decay
            Considers historical updates, leading to better convergence stability
        Disadvantages:
            Hyperparameters are relatively complex
            May require more computational resources
        Application Scenarios:
            High-precision training
            Avoiding overly fast convergence

'''

import torch
from torch import optim
import matplotlib.pyplot as plt

#1. Define function, demonstrate Step interval learning rate decay
def dm01():
    #1. define variable, record initial learning rate, training epoch, number of batches for each epoch
    lr, epoch, iteration = 0.1, 200 ,10

    #2. create dataset, y_true, x,w
    #true value
    y_true = torch.tensor([0])
    #input feature
    x = torch.tensor([1.0], dtype=torch.float32)
    #weight
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    #3. create optimizer object, momentum -> accelerate the convergence of the model and reduce the magnitude of oscillations.
    #param1: parameters to be optimized, param2: learning rate, param3: momentum parameter
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

    #4. create learning rate decay object
    #param1: optimizer object, param2: number of intervals (how many epochs before adjusting the learning rate), param3: learning rate decay coefficient: lr_new = lr_old * gammma
    #Create an evenly spaced learning rate decay scheduler.Create a learning rate scheduler with uniform decay intervals. optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    #5. create two lists to record the learning rate and learning rate for each epoch
    #epoch_list = [1,2,3,...,200]
    #lr_list = [0.1, 0.1, 0.1,... 0.05(50),..0.025(100) ...,0.0125(150)]
    lr_list, epoch_list = [], []

    #6. training loop
    for epoch in range(epoch): #epoc: 0-199
        #7. Get the current epoch and learning rate and store them in a list.
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        #8.Iterate through each epoch and batch for training.
        for batch in range(iteration):
            #9. compute predction and loss base on loss function
            y_pred = x * w 
            #10. compute loss
            loss = (y_true - y_pred)**2
            #11. zero gradient + backpropagation + update parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #12. update learning rate
        scheduler.step()
    #13. print
    print(f'lr_list: {lr_list}')
    #14. plot learning rate decay curve
    #x-axis: epoch, y-axis: learning rate
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

#2. Define function, demontrate Specified interval learning rate decay
def dm02():
    #1. define variable, record initial learning rate, training epoch, number of batches for each epoch
    lr, epoch, iteration = 0.1, 200 ,10

    #2. create dataset, y_true, x,w
    #true value
    y_true = torch.tensor([0])
    #input feature
    x = torch.tensor([1.0], dtype=torch.float32)
    #weight
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    #3. create optimizer object, momentum -> accelerate the convergence of the model and reduce the magnitude of oscillations.
    #param1: parameters to be optimized, param2: learning rate, param3: momentum parameter
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

    #4. create learning rate decay object
    #param1: optimizer object, param2: milestones param3: learning rate decay coefficient: lr_new = lr_old * gammma
    #Create a learning rate scheduler with a specified decay interval.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    milestones = [50, 100, 150]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    #5. create two lists to record the learning rate and learning rate for each epoch
    #epoch_list = [1,2,3,...,200]
    #lr_list = [0.1, 0.1, 0.1,... 0.05(50),..0.025(100) ...,0.0125(150)]
    lr_list, epoch_list = [], []

    #6. training loop
    for epoch in range(epoch): #epoc: 0-199
        #7. Get the current epoch and learning rate and store them in a list.
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        #8.Iterate through each epoch and batch for training.
        for batch in range(iteration):
            #9. compute predction and loss base on loss function
            y_pred = x * w 
            #10. compute loss
            loss = (y_true - y_pred)**2
            #11. zero gradient + backpropagation + update parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #12. update learning rate
        scheduler.step()
    #13. print
    print(f'lr_list: {lr_list}')
    #14. plot learning rate decay curve
    #x-axis: epoch, y-axis: learning rate
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

#3. Define function, demontrate Exponential learning rate decay
def dm03():
    #1. define variable, record initial learning rate, training epoch, number of batches for each epoch
    lr, epoch, iteration = 0.1, 200 ,10

    #2. create dataset, y_true, x,w
    #true value
    y_true = torch.tensor([0])
    #input feature
    x = torch.tensor([1.0], dtype=torch.float32)
    #weight
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    #3. create optimizer object, momentum -> accelerate the convergence of the model and reduce the magnitude of oscillations.
    #param1: parameters to be optimized, param2: learning rate, param3: momentum parameter
    optimizer = optim.SGD(params=[w], lr=lr, momentum=0.9)

    #4. create learning rate decay object
    #param1: optimizer object, param2: milestones param3: learning rate decay coefficient: lr_new = lr_old * gammma
    #create Exponential learning rate decay
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    #5. create two lists to record the learning rate and learning rate for each epoch
    #epoch_list = [1,2,3,...,200]
    #lr_list = [0.1, 0.1, 0.1,... 0.05(50),..0.025(100) ...,0.0125(150)]
    lr_list, epoch_list = [], []

    #6. training loop
    for epoch in range(epoch): #epoc: 0-199
        #7. Get the current epoch and learning rate and store them in a list.
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        #8.Iterate through each epoch and batch for training.
        for batch in range(iteration):
            #9. compute predction and loss base on loss function
            y_pred = x * w 
            #10. compute loss
            loss = (y_true - y_pred)**2
            #11. zero gradient + backpropagation + update parameter
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #12. update learning rate
        scheduler.step()
    #13. print
    print(f'lr_list: {lr_list}')
    #14. plot learning rate decay curve
    #x-axis: epoch, y-axis: learning rate
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

#4. Test
if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()