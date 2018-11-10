import torch
import torchvision
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import numpy as np

from utils import *
from classifier import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--logInt', type=int, default=10)
parser.add_argument('--momentum', type=float, default=.5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--dataPath', type=str, default="data")
parser.add_argument('--msp', dest = 'modelStorePath', type=str, default="savedModels")
parser.add_argument('--vt', dest = 'varThresh', type=float, default=.1)
parser.add_argument('--ns', dest = 'numSamp', type=int, default=20)
parser.add_argument('--tr', dest = 'train', type=int, default=20)
parser.add_argument('--lm', dest = 'loadModel', type=int, default=0)
parser.add_argument('--m', dest = 'model', type=int, default=2)
parser.add_argument('--pt', dest = 'pThresh', type=float, default=.5)
parser.add_argument('--vl', dest = 'val', type=int, default=1)




args = parser.parse_args()
print(args)

random_seed = 1
torch.manual_seed(random_seed)

if not os.path.exists(args.modelStorePath):
    os.makedirs(args.modelStorePath)

train_loader, val_loader, test_loader = getData(args)

if args.model == 1:
    network = Net1()
elif args.model == 2:
    network = Net2()
elif args.model == 3:
    network = Net3()
else:
    print("Error, wrong model specified")
    exit()

if torch.cuda.is_available():
    network = network.cuda()
if args.train==0 or args.loadModel:
  network.load_state_dict(torch.load(args.modelStorePath + '/model' + '_' + str(args.model) + '.pt'))

optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum)

train_losses, train_counter, test_losses, test_counter = [], [], [], []


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):

    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch_idx % args.logInt == 0:
      prettyPrint(epoch, batch_idx, len(data), train_loader, loss.item())
      train_losses.append(loss.item())
      train_counter.append((batch_idx*args.batchSize) + ((epoch-1)*len(train_loader)))

    # break

def test(epoch, bestFP, data_loader):
  network.eval()
  test_loss, correct, outputAll, targetAll = 0, 0, [], []
  with torch.no_grad():
    for data, target in data_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()

      outputAll.append(predHot(output))
      targetAll.append(np.eye(10)[target.data.cpu().numpy()])

  test_loss /= len(data_loader.dataset)
  falsePos = prettyPrint2(test_loss, correct, len(data_loader.dataset), np.concatenate(outputAll, axis=0), np.concatenate(targetAll, axis=0))
  test_losses.append(test_loss)
  test_counter.append(epoch*len(train_loader.dataset))
  makeCF(np.concatenate(targetAll, axis=0), np.concatenate(outputAll, axis=0))

  if falsePos < bestFP:
    bestFP = falsePos
    torch.save(network.state_dict(), args.modelStorePath + '/model_' + str(args.model) + '.pt')
    print("---------------------SAVED------------------")

  return bestFP


def decision(data_loader):
    finOutAll, finTarAll, finleft, finpred, lO = [], [], [], [], 0
    for data, target in data_loader:
      outs = []
      for i in range(args.numSamp):
        outs.append(network(data, decision=1).view(1, -1, 10).data.cpu().numpy())
      
      outs = np.concatenate(outs, axis=0)
      outM = np.mean(outs, axis=0)
      outV = np.var(outs, axis=0)

      network.eval()
      out = np.argmax(predHot(network(data)).tolist(), axis=1)
      network.train()
      target = target.data.cpu().numpy().tolist()

      finOut = []
      finTar = []
      leftOut = []
      predicted = []

      for ind, (i,j) in enumerate(zip(out, target)):
        if outV[ind, i] < args.varThresh:# and outM[ind, i] > args.pThresh:
        # if outM[ind, i] > .5:

          finOut.append(i)
          finTar.append(j)
          predicted.append(ind)
        else:
          lO += 1
          leftOut.append(ind)
      
        # if outM[ind, i] < .5:
        
        if i!=j:
          if ind in predicted:
            print(outM[ind, i])
            print("----------------")

            for k, p in zip(outV[ind].tolist(), outM[ind].tolist()):
              print(k, p)
            print(j)

      finOutAll.append(np.eye(10)[finOut])
      finTarAll.append(np.eye(10)[finTar])
      finleft.append(leftOut)
      finpred.append(predicted)
      
    ratesMC(np.concatenate(finOutAll, axis=0), np.concatenate(finTarAll, axis=0))
    print("Coverage = " + str(100 - float((100.0*lO)/len(data_loader.dataset))))
    np.save('leftOut', np.array(finleft))
    np.save('predicted', np.array(finpred))


bestFP = float('inf')
if args.train:
  for epoch in range(1, args.n_epochs + 1):
    train(epoch)
    if args.val == 1:
      bestFP = test(epoch, bestFP, val_loader)
    else:
      bestFP = test(epoch, bestFP, test_loader)
    genMyPlots(train_losses, test_losses, train_counter, test_counter)

else:
  if args.val == 1:
    decision(val_loader)
  else:
    decision(test_loader)
