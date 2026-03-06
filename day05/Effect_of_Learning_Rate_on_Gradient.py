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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    #5. create two lists to record the learning rate and learning rate for each epoch
    #epoch_list = [1,2,3,...,200]
    #lr_list = [0.1, 0.1, 0.1,... 0.05(50),..0.025(100)...,0.0125(150)]
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

#3. Define function, demontrate Exponential learning rate decay

#4. Test
if __name__ == '__main__':
    dm01()