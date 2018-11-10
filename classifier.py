import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os


# class Conv2dSame(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
#         super().__init__()
#         ka = kernel_size // 2
#         kb = ka - 1 if kernel_size % 2 == 0 else ka
#         self.net = torch.nn.Sequential(
#             padding_layer((ka,kb,ka,kb)),
#             torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
#         )
#     def forward(self, x):
#         return self.net(x)

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self, drp).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net2(nn.Module):
    def __init__(self, drp):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv2_drop = nn.Dropout2d(drp)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64,kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64,kernel_size=5,stride=2)
        self.bn6 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64,128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc_drop = nn.Dropout(drp)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, decision=0):

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.conv2_drop(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.conv2_drop(x)


        x = x.view(x.shape[0], -1)
        # import pdb
        # pdb.set_trace()
        x = F.relu(self.fc1(x))
        x = self.bn7(x)
        x = self.fc_drop(x)
        if decision:
            x = F.softmax(self.fc2(x), -1)
        else:
            x = F.log_softmax(self.fc2(x), -1)

        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self, drp).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv2_drop = nn.Dropout2d(0.25)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64,kernel_size=3)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(576,256)
        self.fc_drop = nn.Dropout(.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, decision=0):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool3(x))
        x = self.conv2_drop(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.pool6(x))
        x = self.conv2_drop(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        if decision:
            x = F.softmax(self.fc2(x), -1)
        else:
            x = F.log_softmax(self.fc2(x), -1)

        return x
