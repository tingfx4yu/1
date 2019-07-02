import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append("game/")
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2

# if gpu is to be used #

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

Tensor = FloatTensor
'''
def plot_graph(mean_reward_list):
    plt.figure(0)
    plt.clf()
    plt.title('Episode Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # 最近100个episode的total reward的平均值 #
    plt.plot(mean_reward_list)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
'''
def ob_process(frame):
    '''
    Parameters
    ----------
    frame: {ndarray} of shape (_,_,3)

    Returns
    -------
    frame: {Tensor} of shape torch.Size([1,84,84])
    '''
    #frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype('float64')
    ret,img=cv2.threshold(frame,1,255,cv2.THRESH_BINARY)
    img=torch.from_numpy(img)
    img=img.unsqueeze(0).type(Tensor)
    return img
class Camera_Agent(nn.Module):
    def __init__(self,ACTION_NUM):
        super(Camera_Agent,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=12*12*64,out_features=256)
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
    def init(self,obs):
        self.num_agents = len(obs['image'])

    def get_obs_cnn(self, obs):
        temp = []
        for i in range(len(obs["image"])):
            temp.append(np.r_[obs["image"][i]])
        temp = np.r_[temp]
        t = np.transpose(temp, (0,3,1,2)).astype(float)
        t = ob_process(t)
        #t /= 255.0
        return t

    def select_action(self,obs,epsilon):
        x = self.get_obs_cnn(obs)
        x = Variable(x.unsqueeze(0))
        x = self.forward(x)
        action_index = x.data.max(1)[1][0]
        if random.random() < epsilon:
            action_index = random.randint(0,7)
        else:
            action_index = x.data.max(1)[1][0]
        return action_index
    def update(self,samples,learn_rate,target_net,BATCH_SIZE,GAMMA):
        obs4_batch = Variable(torch.cat(samples.obs4))
        next_obs4 = Variable(torch.cat(samples.next_obs4))
        action_batch = Variable(torch.cat(samples.act))
        done_batch = samples.done
        reward_batch = torch.cat(samples.reward)
        q_value_batch = target_net(next_obs4).detach()
        target = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
        for i in range(BATCH_SIZE):
            if done_batch[i] == False:
                target[i] = reward_batch[i] + GAMMA * torch.max(q_value_batch.data[i])
            elif done_batch[i] == True:
                target[i] = reward_batch[i]
        action_idx_batch = action_batch.max(1)[1].unsqueeze(1)
        output = self.forward(obs4_batch)
        q_action = output.gather(1,action_idx_batch)
        net_output = q_action.squeeze(1)
        optimizer=torch.optim.RMSprop(self.parameters(),lr=0.00025,alpha=0.95,eps=0.01)
        #loss=criterion(net_output,target)
        ########
        criterion=nn.SmoothL1Loss()
        loss=criterion(input=net_output,target=target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        torch.nn.utils.clip_grad_norm(parameters=self.parameters(), max_norm=1)

        optimizer.step()
class replay_memory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',['obs4','act','next_obs4','reward','done'])
    def __len__(self):
        return len(self.memory)
    def add(self,*args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position+1) % self.capacity
    def sample(self,batch_size):
        batch = random.sample(self.memory,batch_size)
        batch_zip = self.Transition(*zip(*batch))
        return batch_zip
def learn(obs,next_obs,reward,done, 
        REPLAY_MEMORY_CAPACITY,
        BATCH_SIZE,
        LEARNING_RATE,
        GAMMA,
        NET_COPY_STEP,
        OBSERVE,
        TRAIN_FREQ,
        ):
        value_net = Camera_Agent(8).cuda()
        target_net = Camera_Agent(8).cuda()
        buffer = replay_memory(REPLAY_MEMORY_CAPACITY)
        obs = ob_process(obs)
        obs4 = torch.cat([obs,obs,obs,obs])
        act = value_net.select_action(obs4)
        episode_total_reward = 0
        epi_total_reward_list=[]
        mean_reward_list=[]
        # counters #
        time_step=0
        update_times=0
        episode_num=0
        obs_next = ob_process(obs_next)
        obs4_next = torch.cat(([obs4[1:,:,:],obs_next]))
        buffer.add(obs4.unsqueeze(0),act.unsqueeze(0),obs4.unsqueeze(0),Tensor([reward]).unsqueeze(0),done)
        episode_total_reward += reward
        obs4 = obs4_next
        if done.item(0) == False:
            time_step += 1
        elif done.item(0) == True:
            episode_num += 1
            # plot graph #
            epi_total_reward_list.append(episode_total_reward)
            mean100=np.mean(epi_total_reward_list[-101:-1])
            mean_reward_list.append(mean100)
            #plot_graph(mean_reward_list)
            print('episode %d timestep %d : threshold=%.5f total reward=%.2f'%(episode_num,time_step,threshold,episode_total_reward))
            episode_total_reward = 0
        ### do one step update ###
        if (time_step >= OBSERVE) and (time_step % TRAIN_FREQ == 0):
            batch_transition = buffer.sample(BATCH_SIZE)
            '''{Transition}
            0:{tuple} of {Tensor}-shape-torch.Size([4,84,84])
            1:{tuple} of {Tensor}-shape-torch.Size([6])
            2:{tuple} of {Tensor}-shape-torch.Size([4,84,84])
            3:{tuple} of {int}   
            4:{tuple} of {bool}        
            '''
            value_net.update(samples=batch_transition,learn_rate=LEARNING_RATE,
                             target_net=target_net, BATCH_SIZE=BATCH_SIZE,
                             GAMMA=GAMMA)
            update_times += 1

            ### copy value net parameters to target net ###
            if update_times % NET_COPY_STEP == 0:
                target_net.load_state_dict(value_net.state_dict())






        
        



