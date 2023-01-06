import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam 
from torch.optim import SGD 
from torchvision.models import resnet18

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train and test data directory
train_data_dir = "data/train"
test_data_dir = "data/val"


image_size = (128, 128)
mean = np.array([0.5])
std = np.array([0.5])

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.Resize(size=image_size),
    transforms.Normalize(mean, std)
])

val_trasform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize(size=image_size),
    transforms.Normalize(mean, std)
])
    
# hyper-parameters
num_epochs = 10
batch_size = 4
learning_rate = 0.00001

train_dataset = ImageFolder(root=train_data_dir,transform=train_transform)
val_dataset = ImageFolder(root=test_data_dir, transform=val_trasform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#test loader



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
classes = ('no tumor', 'tumor')

criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr = learning_rate)

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

        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
    
    model.train()

print('Finished Training')

# PATH = 'model'
# torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0] * len(classes)
    n_class_samples = [0] * len(classes)

    for images, labels in val_loader: # TODO: zmenit na test loader
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    model.eval()
    

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.4f}%')

    # for i in range(10):
    #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #     print(f'Accuracy of {classes[i]}: {acc} %')