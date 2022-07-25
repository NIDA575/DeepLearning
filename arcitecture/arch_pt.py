import torch.nn as nn

class arch30(nn.Module):

    def __init__(self):
        super(arch30, self).__init__()


    def forward(self, x):
        return x

from torchsummary import summary
import torch
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(224, 224,3)
model = arch30()
print(summary(model, (160,160,3)))

