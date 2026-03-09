"""
Example:
ANN (Artificial Neural Network) case: Mobile phone price classification.

Background:
Based on 20 features of mobile phones, predict the price range of the phone
(4 price categories). This task can be solved using machine learning
or deep learning (recommended).
"""
import torch
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    #5. convert data to tensor and create dataset and dataloader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    #print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')

    #6. return                    20: serve as the number of input features, 4: serve as the number of input features
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))



#todo 5: test
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}')
    print(f'input_dim: {input_dim}, output_dim: {output_dim}')