import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

class arch30(nn.Module):

    def __init__(self):
        super(arch30, self).__init__()
        self.BN = [4, 8, 12, 16, 20, 24, 30, 34]
        self.filters = [32, 64, 128, 256, 512, 1024]
        self.RSK = [
                [(0,2),   [(2,5), (2,3), (1,1)]], #32
                [(3,6),   [(1,1), (1,3), (1,1)]], #64
                [(7,13),  [(1,3), (1,3), (1,3), (1,3), (2,3)]], #128
                [(14,22), [(1,3), (1,3), (1,3), (1,3), (1,3), (1,3), (1,3)]], #256
                [(23,31), [(1,3), (2,3), (1,3), (1,3), (1,3), (1,3), (1,3)]], #512
                [(32,38), [(1,3), (1,3), (1,1), (2,1), (2,1), (2,1), (2,1)]], #1024 
                ] #RSK=range, stride, kernel
        self.dense_values = [4196, 1024, 512, 128, 6]
        bn=0
        cn=0
        x=[]
	#self.parameters = nn.ModuleList()
        for i in range(len(self.filters)):
            p = 0
            for j in range(self.RSK[i][0][0],self.RSK[i][0][1]+1):
                #print(range(self.RSK[i][0][0],self.RSK[i][0][1]+1))
                if i==0 and j==0:
                    in_channels = 3
                    padding = 0
                else:
                    padding = 1
                if j not in self.BN:
                    #print(j)
                    #print(self.RSK[i][1][2][0])
                    s_size = self.RSK[i][1][p][0]
                    k_size = self.RSK[i][1][p][1]
                    globals()['self.conv%s' % cn ] = nn.Conv2d(in_channels, out_channels=self.filters[i], kernel_size=k_size, stride=s_size, padding=padding)
                    print(['self.conv%s' % cn])
                    y=globals()['self.conv%s' % cn ]
                    x.append(y)
                    in_channels = self.filters[i]
                    p = p+1
                    cn = cn+1
                else:
                    try:
                        globals()['self.bn%s' % bn] = nn.BatchNorm2d(self.filters[i])
                        y=globals()['self.bn%s' % bn ]
                        print(['self.bn%s' % bn])
                        x.append(y)
                    except:
                        try:    
                            globals()['self.bn2_%s' % bn] = nn.BatchNorm2d(self.filters[i]*2)
                            y=globals()['self.bn2_%s' % bn ]
                            print(['self.bn2_%s' % bn])
                            x.append(y)
                        except:    
                            globals()['self.bn4_%s' % bn] = nn.BatchNorm2d(self.filters[i]*4)
                            y=globals()['self.bn4_%s' % bn ]
                            print(['self.bn4_%s' % bn])
                            x.append(y)
                    bn=bn+1
        dn=0
        
        for i in range(len(self.dense_values)):
            if i == 0:
                in_features = 16384
                out_features = self.dense_values[i]
            globals()['self.dense%s' % i] = nn.Linear(in_features, self.dense_values[i])
            y=globals()['self.dense%s' % i ]
            x.append(y)
            in_features = self.dense_values[i]
	    #print(globals()['self.dense%s' % i])
        self.relu = nn.ReLU()
        x.append(self.relu)
        self.sigmoid = nn.Sigmoid()
        x.append(self.sigmoid)
        self.flatten = nn.Flatten()
        x.append(self.flatten)
        self.softmax = nn.Softmax(dim=1)
        x.append(self.softmax)
        self.parameters = nn.ModuleList(x)


    def forward(self, x):
        bn=0
        cn=0
        
	   
        for i in range(len(self.filters)):
            p = 0
            for j in range(self.RSK[i][0][0],self.RSK[i][0][1]+1):
                if j not in self.BN:
                    x= self.sigmoid(globals()['self.conv%s' % cn] (x))
                    cn = cn+1
                else:
                    try:
                        x= globals()['self.bn%s' % bn] (x)
                    except:
                        try:
                            print("bn*2")
                            x= globals()['self.bn2_%s' % bn] (x)
                        except:
                            print("bn*4")
                            x= globals()['self.bn4_%s' % bn] (x)
                    bn=bn+1
        dn=0
        x = self.flatten(x)
        x.resize
        print(x.shape)
        for i in range(len(self.dense_values)):
            x = self.relu(globals()['self.dense%s' % i](x))
        x = self.softmax(x)
        
        return x
 

x = torch.randn(1,3,160,160)
model = arch30()
summary(model, (3,160,160), batch_size=1)
print(model(x))
exit(0)
data_dir = './data'

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(160), transforms.ToTensor(), ])    
    test_transforms = transforms.Compose([transforms.Resize(160), transforms.ToTensor(),])    
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    
    
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
model.to(device)
#exit(0)

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device) , labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    print("<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>")
                    print(equals)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'aerialmodel.pth')

