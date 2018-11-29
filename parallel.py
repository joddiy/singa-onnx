import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Parallel(nn.Module):
    def __init__(self):
        super(Parallel, self).__init__()
        self.cnn0 = nn.Conv2d(1, 1, 3, padding=1)
        self.cnn1 = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(18,18)

    def forward(self,x0,x1):
        x0 = self.cnn0(x0)
        x1 = self.cnn0(x1)
        x0 = x0.view(x0.shape[0], -1)
        x1 = x1.view(x0.shape[0], -1)
        x=torch.cat((x0,x1),dim=1)
        x = self.fc(x)
        return x
x0 = torch.zeros((1,1,3,3))
x1 = torch.zeros((1,1,3,3))
model = Parallel()
print(model(x0,x1))