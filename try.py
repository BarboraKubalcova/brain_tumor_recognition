import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam 

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

#train and test data directory
train_data_dir = "data/training"
test_data_dir = "data/testing"


image_size = (128, 128) # resizing data

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.RandomRotation(degrees=15),
    transforms.Resize(size=image_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_trasform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize(size=image_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    

train_dataset = ImageFolder(root=train_data_dir,transform=train_transform)
test_dataset = ImageFolder(root=test_data_dir, transform=test_trasform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



class BrainTumorModel(nn.Module):

  def __init__(self):      
    super(BrainTumorModel, self).__init__()
    # n_out = ((n_in + 2p - k)/ s) + 1 (k - kernel size, p - padding size, s - stride size)
    self.conv1 = nn.Sequential(
        nn.Conv2d(1,256,kernel_size=3), #output from this will be 126*126*256
        nn.MaxPool2d(2,2), # 63*63*256
        nn.Conv2d(256,32,kernel_size=2) #63-2+1 => 62*62*32
    )
    self.linear1 = nn.Linear(62,128)
    self.linear2 = nn.Linear(128,64)
    self.flat = nn.Flatten(1)
    self.linear3 = nn.Linear(126976,2) 


  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    x = self.flat(x)
    x = self.linear3(x)

    return x

model = BrainTumorModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)
epoch_acc = []
# training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')