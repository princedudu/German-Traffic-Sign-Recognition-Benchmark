import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv1_drop = nn.Dropout2d(0.2)
        self.bn1 = nn.BatchNorm2d(20)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.bn2 = nn.BatchNorm2d(50)

        self.conv3 = nn.Conv2d(50, 120, kernel_size=5, padding=1)
        self.conv3_drop = nn.Dropout2d(0.4)
        self.bn3 = nn.BatchNorm2d(120)

        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, nclasses)
        self.fc1 = nn.Linear(1080, 400)
        self.fc2 = nn.Linear(400, 84)
        self.fc3 = nn.Linear(84, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = self.bn1(x)

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn2(x)

        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = self.bn3(x)

        x = x.view(-1, 1080)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)