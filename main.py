import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from sklearn.utils import shuffle
from PIL import Image
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BrainMRIDataset(Dataset):

  def __init__(self, data_dir = "data", reshape=True, height=128, width=128, autoencoder = False):

    self.dataDirectory = data_dir
    self.normal_links = glob(data_dir+'/NORMAL/*')
    self.tumor_links = glob(data_dir+'/TUMOR/*')

    self.height = height
    self.width = width
    self.reshape = reshape
    self.autoencoder = autoencoder

    labels = [0] * len(self.normal_links) + [1] * len(self.tumor_links)

    image_links = self.normal_links + self.tumor_links

    self.dataframe = pd.DataFrame({"image":image_links, "labels":labels})
    self.dataframe.reset_index(inplace = True ,drop=True) # inplace - we are nor saving any variable

  def __len__(self):
    return len(self.normal_links)+len(self.tumor_links)

  def __getitem__(self,idx):

    image_list = self.dataframe["image"][idx]
    label_list = self.dataframe["labels"][idx]

    if type(image_list) == str:
      image_list = [image_list]

    if not isinstance(label_list,np.int64):
      label_list = label_list.values

    image_array = []

    for image in image_list:
      image = Image.open(image).convert("L")

      if self.reshape:
        image = image.resize((self.height,self.width))

      array = np.asarray(image)

      array = array.reshape(1,self.height,self.width)

      image_array.append(array)

    return [torch.tensor(image_array,device=device),torch.tensor(label_list,device=device)]

  def __repr__(self):
    return str(self.dataframe.head(10))

dataset = BrainMRIDataset()


class BrainTumorModel(nn.Module):

  def __init__(self):      
    super().__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(1,256,kernel_size=3), #126*126*256
        nn.MaxPool2d(2,2), # 63*63*256

        nn.Conv2d(256,32,kernel_size=2) #63-2+1 = 62*62*32
    )

    # n-f+2p/s +1 

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
     
model = BrainTumorModel()
model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs =5
batch_size = 32
loss_list = []

for epoch in range(epochs):
  total_loss = 0.0

  for n in range(len(dataset)//batch_size):

    data , target = dataset[n*batch_size : (n+1)*batch_size]

    ypred = model.forward(data.float())
    loss = loss_fn(ypred,target)

    total_loss+=loss

    optimizer.zero_grad() #clear the gradients
    loss.backward() # calculate the gradeint
    optimizer.step() # Wn = Wo - lr* gradeint

  loss_list.append(total_loss/batch_size)

  print("Epochs {}  Training Loss {:.2f}".format(epoch+1,total_loss/n))




