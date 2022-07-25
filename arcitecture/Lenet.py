import torch
from torch import nn

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.softmax = nn.Softmax(dim =1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

if __name__ == "__main__":
    x = torch.randn(64, 1, 32, 32)
    model = Lenet()
    print(model(x).shape)

