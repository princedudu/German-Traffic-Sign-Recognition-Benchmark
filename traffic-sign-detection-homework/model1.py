import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv1 = nn.Conv2d(3, 108, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(108, 200, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, nclasses)
        self.fc1 = nn.Linear(5000, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 5000)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)