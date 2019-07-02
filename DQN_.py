import torch
import torch.nn as nn
import torch.nn.functional as F

class dqn_net_state(nn.Module):
    def __init__(self,ACTION_NUM):
        super(dqn_net_state,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=7*7*64,out_features=256)
        #self.fc1 = nn.Linear(in_features=9*9*32, out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=ACTION_NUM)
        self.action_num=ACTION_NUM
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        output=self.fc2(output)
        return output

class dqn_net_oth(nn.Module):
    def __init__(self,ACTION_NUM,STATE_OTH):
        super(dqn_net_oth,self).__init__()
        self.fc1 = nn.Linear(STATE_OTH,32)
        self.fc2 = nn.Linear(32,ACTION_NUM)
    def forward(self,input,ACTION_NUM,STATE_OTH):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x
