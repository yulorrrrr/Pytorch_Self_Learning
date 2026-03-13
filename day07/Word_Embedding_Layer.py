"""
Example:
Demonstrate the API usage of the word embedding layer.

RNN Introduction:
Full name: Recurrent Neural Network, mainly used to process sequential data.

Sequential data: later data depends on earlier data.
Examples: weather forecasting, stock market analysis, text generation...

Components:
Word embedding layer
Recurrent neural network layer
Output layer

Introduction to the word embedding layer (function):
Convert words (or the indices corresponding to words) into word vectors.
"""

import torch
import jieba
import torch.nn as nn

#1. define function to demenstrate word embedding later's API （Word index → word embedding）
def dm01():
    #1. define a sentence
    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
    #2. use jieba for words segmentation
    words = jieba.lcut(text)
    print(words)

    #3. create word embedding layer
    #param 1: the length of words, param 2: the dimension for word vector
    embed = nn.Embedding(len(words), 10)

    #4. get the index for each word object
    #enumerate: return the value and the index for each value in the list
    for i, word in enumerate(words):
        #print(i, word)

        #5. change word index(tensor) to word vector
        word_vector = embed(torch.tensor(i)) #random

        print(f'word:{word}, \nword vector:{word_vector}\n')





#2. test
if __name__ == '__main__':
    dm01()