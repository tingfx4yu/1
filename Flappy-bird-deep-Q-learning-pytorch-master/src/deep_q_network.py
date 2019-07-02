"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
from torch.autograd import Variable
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        action_prob=F.softmax(self.fc2(output),1)
        return action_prob

        return output

if __name__ == '__main__':
    net = DeepQNetwork()
    data = Variable(torch.randn([1,128,128]))
    state = torch.cat(tuple(data for _ in range(4)))[None, :, :, :]
    action_prob = net(state)
    c = Categorical(action_prob)
    action = c.sample()
    print('action item = ',action.item(),'action_prob = ',action_prob[:,action.item()].item())

    print(net(state))
    #print(net(state).squeeze(0))
    #print(net(state)[0])