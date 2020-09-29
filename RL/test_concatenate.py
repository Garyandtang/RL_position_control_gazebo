'''
this works with output
>>> output
tensor([[ 0.4694, -0.0458]], grad_fn=<CatBackward>)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
class ActorBlock(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorBlock, self).__init__()
        self.linear_v = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512,1),
                nn.Sigmoid()
                )
        self.angular_v = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512,1),
                nn.Tanh()
                )
    def forward(self, state):
        x1 = self.linear_v(state)
        x2 = self.angular_v(state)
        x = torch.cat((x1, x2), dim=1)
        return x
        

model = ActorBlock(4,2)
state = np.array([1,1,1,1])
state = torch.FloatTensor(state.reshape(1, -1))
output = model(state)

# batch_size = 2
# image = torch.randn(batch_size, 3, 299, 299)
# data = torch.randn(batch_size, 10)

# output = model(image, data)
