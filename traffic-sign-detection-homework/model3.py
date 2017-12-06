import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding = 2)
        self.conv1_drop = nn.Dropout2d(0.1)
        self.conv2_drop = nn.Dropout2d(0.2)
        self.conv3_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(3584, 1024)
        # self.fc_drop = nn.Dropout()
        self.fc2 = nn.Linear(1024, nclasses)

    def forward(self, x):
        x1 = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        x3 = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x2)), 2))
        x1 = F.max_pool2d(x1, 4).view(-1, 4*4*32)
        x2 = F.max_pool2d(x2, 2).view(-1, 4*4*64)
        x3 = x3.view(-1, 4*4*128)
        x = torch.cat((x1, x2, x3), 1)
        x = x.view(-1, 3584)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
