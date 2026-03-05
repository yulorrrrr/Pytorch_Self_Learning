'''
Regression problem loss function:
    MAE: Mean Absolute Error (L1Loss)
        formula: 
            Sum of absolute errors / the total number of samples

        Similar to L1 regularization, it can drive some weights to zero, resulting in sparse parameters.
        It is not smooth at zero, which may cause the optimizer to miss the minimum.
    
    MSE: Mean Squared Error;
        formula:
            Sum of squared errors / the total number of samples

        Penalizes large errors more heavily because the error is squared.
        Smooth and differentiable everywhere, which makes optimization easier.
        Sensitive to outliers since large errors grow quadratically.

    smooth L1:
        formula:  
            If |x| < 1: 0.5 * x²
            If |x| ≥ 1: |x| - 0.5
        
        Combines the advantages of L1 loss and L2 loss.
        Behaves like MSE for small errors and like MAE for large errors.
        Less sensitive to outliers than MSE while remaining smooth near zero.

'''
import torch
import torch.nn as nn

#1. Define function, MASloss function
def dm01():
    #1. create true value
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    #2. create predict value
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    #3. create MAE Loss function object
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()

    #4. compute loss
    loss = criterion(y_pred, y_true)

    #5. print
    print(f'MAE: {loss}')

#2. Define function, MSEloss function
def dm02():
    #1. create true value
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    #2. create predict value
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    #3. createSE Loss function object
    criterion = nn.MSELoss()

    #4. compute loss
    loss = criterion(y_pred, y_true)

    #5. print
    print(f'MSE: {loss}')

#3. Define function, Smooth L1 loss function
def dm03():
    #1. create true value
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    #2. create predict value
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)

    #3. createSE Loss function object
    criterion = nn.SmoothL1Loss()

    #4. compute loss
    loss = criterion(y_pred, y_true)

    #5. print
    print(f'SmoothL1: {loss}')


#4. Test
if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()