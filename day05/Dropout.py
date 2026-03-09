'''
Function of Regularization: Mitigate Model Overfitting

Methods of Regularization
    L1 Regularization: 
        Weights can become 0, which is equivalent to feature selection (dimension reduction).
    L2 Regularization:
        Weights can approach 0 infinitely, making the model smoother.
    Dropout:
    Random deactivation.
    During each training iteration, randomly deactivate some neurons to prevent certain features 
    from having excessive influence on the results, thereby reducing overfitting.
    BN (Batch Normalization):

'''

import torch
import torch.nn as nn

#1.Define function, demontrate Dropout
def dm01():
    #1. create hidden layer output 
    t1 = torch.randint(0, 10, size=(1,4)).float()
    print(f't1: {t1}')

    #2.Compute the next weighted sum and apply the activation function.
    #2.1 Create a fully connected layer (serving as a linear layer).
    #param1: input feature dimension, param2: output feature dimension
    Linear1 = nn.Linear(4, 5)

    #2.2. weighted sum 
    l1 = Linear1(t1)
    print(f'l1: {l1}')

    #2.3 activation function
    output = torch.relu(l1) 
    print(f'output: {output}')

    #3. apply dropout to the activations -> Only applied during training, not during testing.
    dropout = nn.Dropout(p=0.5) #p: dropout rate, the probability of deactivating a neuron
    #The specific dropout operation
    d1 = dropout(output)
    print(f'd1: {d1}')



#2. Test
if __name__ == '__main__':
    dm01()