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



class MNISTJ(torch.utils.data.Dataset):

  def __init__(self, dataPath, transform=None, train=True):
        'Initialization'
        self.dataPath = dataPath
        self.transform = transform
        self.train = train

        if train:
            self.data = torch.from_numpy(np.load('mnistTr.npy'))
        else:
            self.data = torch.from_numpy(np.load('mnistTe.npy'))

        # import pdb
        # pdb.set_trace()

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        if torch.cuda.is_available():
            typ = torch.cuda.FloatTensor
            typ2 = torch.cuda.LongTensor
        else:
            typ = torch.FloatTensor
            typ2 = torch.LongTensor

        X = self.data[index, 1:].view(1, 28, 28).type(typ)
        y = self.data[index, 0].type(typ2)

        return X, y

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)

def getData(args):

    # train_loader = torch.utils.data.DataLoader(
    # torchvision.datasets.MNIST(args.dataPath, train=True, download=True,
    #                             transform=torchvision.transforms.Compose([
    #                             torchvision.transforms.ToTensor(),
    #                             torchvision.transforms.Normalize(
    #                                 (0.1307,), (0.3081,))
    #                             ])),
    # batch_size=args.batchSize, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
                                MNISTJ(args.dataPath, train=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.RandomRotation(degrees=20), 
                                RandomShift(3),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=args.batchSize, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                                MNISTJ(args.dataPath, train=False,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=1000, shuffle=True)

    return train_loader, test_loader

def prettyPrint(epoch, batch_idx, batchSize, train_loader, loss):

  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    epoch, batch_idx * batchSize, len(train_loader.dataset),
    100. * batch_idx / len(train_loader), loss))

def prettyPrint2(test_loss, correct, testSize, output, target):
    
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, testSize, 100. * correct / testSize))

    allRates = np.array(ratesMC(output, target))

    return allRates[:,2].mean()
def rates(pred_labels, true_labels):
  TP = 100*np.sum(np.logical_and(pred_labels == 1, true_labels == 1))/(1.0*np.sum(true_labels))
  TN = 100*np.sum(np.logical_and(pred_labels == 0, true_labels == 0))/(-1.0*np.sum(true_labels-1))
  FP = 100*np.sum(np.logical_and(pred_labels == 1, true_labels == 0))/(-1.0*np.sum(true_labels-1))
  FN = 100*np.sum(np.logical_and(pred_labels == 0, true_labels == 1))/(1.0*np.sum(true_labels))

  return [TP, TN, FP, FN]
def ratesMC(pred_labels, true_labels):
    allRates = []
    n = pred_labels.shape[0]
    name = ['True Pos.', 'True Neg.', 'False Pos.', 'False Neg']
    for i in range(10):
        allRates.append(rates(pred_labels[:,i], true_labels[:,i]))

        out = ""
        for j in range(4):
            try:
                out += " " + name[j] + ": " + str(allRates[i][j])
            except:
                import pdb
                pdb.set_trace()

        print(out)
    
    allRates = np.array(allRates)
    out = "=========== MEAN ==============\n"
    for j in range(4):
        out += " " + name[j] + ": " + str(allRates[:,j].mean())
    print(out)
    return allRates





def genMyPlots(train_losses, test_losses, train_counter, test_counter):
  fig = plt.figure()
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.savefig('Loss.png')

def predHot(output):
  a = output.data.cpu().numpy()
  b = np.zeros_like(a)
  b[np.arange(len(a)), a.argmax(1)] = 1

  return b

def makeCF(y_test, y_pred):
    array = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    df_cm = pd.DataFrame(array, index = [i for i in "0123456789"],
                    columns = [i for i in "0123456789"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('CF.png')