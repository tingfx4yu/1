import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def ob_process(img):
    img=torch.FloatTensor(torch.from_numpy(img)).cuda()
    img=img.unsqueeze(0)
    return img

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=12*12*64,out_features=256)
        #self.fc1 = nn.Linear(in_features=9*9*32, out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=8)
        #self.action_num=ACTION_NUM
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        output=self.fc2(output)
        return output
       
class replay_memory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque(maxlen = capacity)
        self.position = 0
        self.Transition = namedtuple('Transition',['state_cnn','act','reward','done'])
    def __len__(self):
        return len(self.memory)
    def add(self,state_cnn,act,reward,done):
        e = self.Transition(state_cnn,act,reward,done)
        self.memory.append(e)
    def sample(self,batch_size):
        rand_idx = np.random.randint(1,len(self.memory)-1,batch_size)
        next_rand_idx = rand_idx + 1
        state = [self.memory[i] for i in rand_idx]
        next_state = [self.memory[i] for i in next_rand_idx]

        state_cnn = torch.cat([e.state_cnn for e in state if e is not None])
        next_state_cnn = torch.cat([e.state_cnn for e in next_state if e is not None])
        #action = torch.cat([e.act for e in state if e is not None])
        
        action = torch.from_numpy(np.vstack([e.act for e in state if e is not None])).float().cuda()
        reward = torch.from_numpy(np.vstack([e.reward for e in state if e is not None])).float()
        done = [e.done for e in state if e is not None]
        return state_cnn, next_state_cnn, action, reward, done

class Agent():
    def __init__(self):
        super(Agent,self).__init__()
        self.value_net,self.target_net = DQN().cuda(), DQN().cuda()
        self.num_agents = 9
        self.step_counter = 0
        self.buffer = replay_memory(5000)
        self.batch_size = 32
        self.exploration_step = 0

    def get_obs_cnn(self,obs):
        temp = []
        for i in range(self.num_agents):
            temp.append(np.r_[obs["image"][i]])
        #print(temp)
        temp = np.r_[temp]
        #print(temp)
        t = ob_process(temp)
        #print(t.size())
        return t

    def select_action(self,obs,epsilon):
        state = self.get_obs_cnn(obs)
        action_value = np.zeros(9,)
        action_index = np.zeros(9)
        for i in range(self.num_agents):
            if np.random.randn < epsilon:
                action_index[i] = random.randint(0,7)
            else:
                action_value[i] = self.value_net.forward(torch.cat(state[i],state[i],state[i],state[i]))
                action_index_agent = action_index[i].data.max(1)[1][0]
                action_index[i] = action_index_agent
        return action_value, action_index

    
    def learn(self):
        self.step_counter += 1
        self.GAMMA = 0.99
        GAMMA = self.GAMMA
        state_cnn,next_state_cnn,action,reward,done = self.buffer.sample(self.batch_size)
        #action = Tensor(action).cuda()
        action.type(torch.cuda.LongTensor)
        #q_value_batch = self.target_net(next_state_cnn).detach()
        target = Variable(torch.zeros(self.batch_size,self.num_agents).type(Tensor))
        ########### 
        for i in range(self.num_agents):
            for j in range(self.batch_size):
                target[j][i] = reward[j][i] + GAMMA * torch.max(self.target_net(torch.cat(next_state_cnn[j][i],next_state_cnn[j][i],next_state_cnn[j][i],next_state_cnn[j][i]).data[j]).detach()
        #target=Variable(torch.zeros(self.num_agents,self.batch_size).type(Tensor))
        out = self.value_net.forward(state_cnn)
        action_idx_batch = out.max(1)[1].unsqueeze(1)
        q_action = out.gather(1,action_idx_batch)
        net_output = q_action
        criterion = nn.MSELoss()
        optimizer=torch.optim.RMSprop(self.value_net.parameters(),lr=0.00025,alpha=0.95,eps=0.01)
        loss = criterion(net_output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if self.step_counter % 10 == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def store_experience(self, obs,action,reward,done):
        state_cnn = self.get_obs_cnn(obs)
        self.buffer.add(state_cnn,action,reward,done)