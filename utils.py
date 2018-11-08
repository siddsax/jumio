import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np

def getData(args):

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(args.modelStorePath, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=args.batchSize, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(args.modelStorePath, train=False, download=True,
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
    print("")

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