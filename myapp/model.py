import os
import torch
import numpy as np
from torch import nn
import pickle
from torchvision import transforms

class CNN_for_EMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            nn.Flatten(),

            nn.LazyLinear(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=128, out_features=47)
            
        )

    def forward(self,x):
        return self.model(x)
    
class Model:
    def __init__(self):
        labels_path = os.path.join('myapp', 'labelz.pickle')
        with open(labels_path, 'rb') as f:
            self.labelz = pickle.load(f)

        self.model = CNN_for_EMNIST()
        
        self.transform = transforms.Compose([
            transforms.Normalize([0.5], [0.5])
        ])
        

    def predict(self, x): 
        data = x.float()
        data = torch.fliplr(data)
        data = torch.rot90(data,1)
        data = data.unsqueeze(0)
        data = data.unsqueeze(0)
       
        data = self.transform(data)
        
        self.model.load_state_dict(torch.load('myapp/mod1608_087_no_aug_no_batch.pth'))
        with torch.no_grad():
            pred = self.model(data)
            _, pred_label = torch.max(pred, 1)
        symbol = self.labelz[int(pred_label)]
        
        return symbol