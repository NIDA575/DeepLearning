import torch
from torch import nn

class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 100)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.sigmoid(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.sigmoid(self.conv3(x))
        print(x.shape)
        x = self.sigmoid(self.conv4(x))
        print(x.shape)
        x = self.sigmoid(self.conv5(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        print(x.shape)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

x = torch.randn(1,3,227,227)
model = Alexnet()
print(model(x).shape)
