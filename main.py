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
from torch.optim import SGD 
from torchvision.models import resnet18
import time as time


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train and val data directory
data_dir = "data"
train_data_dir = "data/train"
val_data_dir = "data/val"
test_data_dir = "data/test"


image_size = (128, 128)
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.Resize(size=image_size),
    transforms.Normalize(mean, std)
])

val_trasform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=image_size),
    transforms.Normalize(mean, std)
])

    
# parameters
num_epochs = 10
batch_size = 4
learning_rate = 0.0001
classes = ['NORMAL', 'TUMOR']

# dataset preparation
train_dataset = ImageFolder(root = train_data_dir,transform = train_transform)
val_dataset = ImageFolder(root = val_data_dir, transform = val_trasform)
test_dataset = ImageFolder(root = val_data_dir, transform = test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model class
class BrainTumorModel(nn.Module):
    def __init__(self):
        super(BrainTumorModel, self).__init__()
        self.rn18 = resnet18(weights='ResNet18_Weights.DEFAULT') 
        self.fc = nn.Linear(in_features=512, out_features=2) # fully-conected layer

    def forward(self, x): 
        x = self.rn18.conv1(x)
        x = self.rn18.bn1(x)
        x = self.rn18.relu(x)
        x = self.rn18.maxpool(x)

        x = self.rn18.layer1(x)
        x = self.rn18.layer2(x)
        x = self.rn18.layer3(x)
        x = self.rn18.layer4(x)

        x = self.rn18.avgpool(x) 
        x = x.view(x.size(0), 512) # x.size(0) - size of the batch

        x = self.fc(x)
        return x
    

# training function 
def train_model(model, criterion, optimizer, n_epochs = num_epochs):
    since = time.time()
    n_total_steps = len(train_loader)
    # training loop
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 25 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
 
    t = time.time() - since
    print(f"Training took {int(t) // 60}m {int(t % 60)}s")
    return model


model = BrainTumorModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr = learning_rate)


model = train_model(model, criterion, optimizer)
model.eval()
print('Finished Training')

epoch_acc = []
# validation loop
for i in range(num_epochs):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images_val, labels_val in val_loader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs = model(images_val)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels_val.size(0)
            n_correct += (predicted == labels_val).sum().item()

        acc = 100.0 * n_correct / n_samples
        epoch_acc.append(acc)


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0] * len(classes)
    n_class_samples = [0] * len(classes)

    for images, labels in test_loader: 
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)): 
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc:.4f}%')

    for i in range(len(classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f}%')



epoch = [x+1 for x in range(num_epochs)]
plt.plot(epoch, epoch_acc)
plt.xlabel("epoch")
plt.ylabel("acc %")
plt.show()


mapping = {0:"NO",1:"Yes"}
fig = plt.figure(figsize=(5,5))

for i, (images, labels) in enumerate(train_loader):
    if i == 5:
        break
    images = images.to(device)
    labels = labels.to(device)
    outputs = model.forward(images)
    _, predicted = torch.max(outputs, 1)
    plt.imshow(images[0][0].cpu())
    plt.title(f"Reality: {mapping[int(labels[0].detach().numpy())]} Predicted: {mapping[int(predicted[0].detach().numpy())]}")
    plt.show()