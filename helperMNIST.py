import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val', dest = 'val', type=int, default=1)
args = parser.parse_args()

data = genfromtxt('mnist.csv', delimiter=',')[1:]

data = np.take(data,np.random.rand(data.shape[0]).argsort(),axis=0,out=data)
data = data[data[:,0].argsort()]

test = []
train = []
val = []
ct = 0
j = 0
for i in range(data.shape[0]):
    if ct >= 1000:
        if ct >=1500 or args.val==0:
            train.append(i)
            ct = 0
            j+=1
        else:
            if j == data[i, 0]:
                val.append(i)
                ct += 1
            else:
                train.append(i)
    else:
        if j == data[i, 0]:
            test.append(i)
            ct += 1
        else:
            train.append(i)


trData = data[train, :]
teData = data[test, :]
vlData = data[val, :]

if not os.path.exists('data'):
    os.makedirs('data')


np.save('data/mnistTr', trData)
np.save('data/mnistTe', teData)
if args.val:
    np.save('data/mnistVl', vlData)
