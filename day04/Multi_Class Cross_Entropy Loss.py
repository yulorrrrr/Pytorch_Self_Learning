'''
Loss function introduction:
    loss fuction, also called cost function, objective function, error function is used to measure 
    how well the model performs (i.e., how well the model fits the data).

Category:
    Classification problem:
        Multi-class Cross-Entropy Loss: CrossEntropyLoss
        Binary Crorss-Entropy Loss: BCELoss

    Regression problem:

        Loss =  ∑ylog(S(f(x)))
        x: sample
        f(x) : weighted sum
        S(f(x)) : predicted probability
        y: real probability
    
    Summarize: The loss function minimizes the negative logarithm of the probability of the correct class.
    ** CrossEntyopyLoss = Softmax() + loss compute, (output layer no need to use softmax() if this loss function is used)

'''
import torch
import torch.nn as nn

#1. Define function, Multi-class Cross-Entropy Loss:
def dm01():
    #1. Create the ground truth labels -> y
    #y_true = torch.tensor([[0,1,0], [0,0,1]], dtype=torch.float)
    y_true = torch.tensor([1,2])

    #2. Create the ground predict labels -> f(x)
    y_pred = torch.tensor([[0.1, 0.8, 0.1],[0.7, 0.2, 0.1]], requires_grad=True, dtype=torch.float)
        
    #3. Create Multi-class Cross-Entropy Loss Function
    criterion = nn.CrossEntropyLoss() #mean loss

    #4. Compute loss value
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')

#2. test
if __name__ == '__main__':
    dm01()