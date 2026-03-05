'''
Binary-Class Cross-Entropy Loss(BCELoss):
    Formula:
        Loss = -ylog(predict value) - (1-y)log(1-predict value)
    ** Sigmoid activation function isn't included in the formula, Sigmoid needs to be specified manually

'''
import torch
import torch.nn as nn

#1. define function
def dm01():
    #1. create true value
    y_true = torch.tensor([0,1,0], dtype=torch.float)

    #2. set predict value
    y_pred = torch.tensor([0.6901, 0.5423, 0.2639])

    #3. create binary class cross entropy loss
    criterion = nn.BCELoss() #mean loss

    #4. compute loss
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

#2. test
if __name__ == '__main__':
    dm01()