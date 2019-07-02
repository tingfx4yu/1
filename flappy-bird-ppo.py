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
import wrapped_flappy_bird as game
import os
import cv2
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import argparse
import pickle
from itertools import count

# if gpu is to be used #

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state','done'])

Tensor = FloatTensor
#########################
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

def ob_process(frame):
    '''
    Parameters
    ----------
    frame: {ndarray} of shape (_,_,3)

    Returns
    -------
    frame: {Tensor} of shape torch.Size([1,84,84])
    '''
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame=frame.astype('float64')
    ret,img=cv2.threshold(frame,1,255,cv2.THRESH_BINARY)
    img=torch.from_numpy(img)
    img=img.unsqueeze(0).type(Tensor)
    return img

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=7*7*64,out_features=256)
        #self.fc1 = nn.Linear(in_features=9*9*32, out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=2)
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        action_prob=F.softmax(self.fc2(output),1)
        return action_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=7*7*64,out_features=256)
        #self.fc1 = nn.Linear(in_features=9*9*32, out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=1)
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        value = self.fc2(output)
        return value
class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 8000
    batch_size = 32
    def __init__(self):
        super(PPO,self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        if torch.cuda.is_available():
            self.actor_net.cuda()
            self.critic_net.cuda()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')
    def select_action(self,state):
        state = Variable(state.unsqueeze(0))
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action_prob.item()].item()

    def get_value(self,state):
        state = Variable(state.unsqueeze(0))
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')
    def store_transition(self,transition):
        self.buffer.append(transition)
        self.counter += 1
    def update(self,gamma):
        state = torch.tensor([t.state for t in self.buffer], dtype = FloatTensor)
        action = torch.tensor([t.action for t in self.buffer], dtype = LongTensor)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype = FloatTensor). view(-1,1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0,R)
        Gt = torch.tensor(Gt, dtype = FloatTensor)
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))),self.batch_size,False):
                #if self.training_step % 1000 = 0:
                Gt_index = Gt[index].view(-1,1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                action_prob = self.actor_net(state[index]).gather(1,action[index])

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)*advantage
                '''
                update actor network
                '''
                action_loss = - torch.min(surr1,surr2).mean()
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                '''
                update critic network
                '''
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
        del self.buffer[:]

def main():
    agent = PPO()
    observe = 100
    action_meaning = [
        np.array([1,0]),
        np.array([0,1])
    ]
    action_space = []
    for i in range(2):
        action_onehot = Tensor([0]*2)
        action_onehot[i] = 1
        action_space.append((i,action_onehot))
    action0 = action_meaning[action_space[0][0]]
    obs,_,_ = env.frame_step(action0)
    obs = ob_process(obs)
    obs4 = torch.cat(([obs,obs,obs,obs]),0)
    episode_total_reward = 0
    epi_total_reward_list=[]
    mean_reward_list=[]
    # counters #
    time_step=0
    update_times=0
    episode_num=0
    while episode_num < 200000:
        action, action_prob = agent.select_action(obs4)
        obs_next, reward, done = env.frame_step(action)
        obs_next = ob_process(obs_next)
        obs4_next = torch.cat(([obs4[1:,:,:],obs_next]),dim = 0)
        trans = Transition(obs4,action,action_prob,reward,obs4_next,done)
        agent.store_transition(trans)
        episode_total_reward += reward

        obs4 = obs4_next

        if done == False:
            time_step += 1
        elif done == True:
            episode_num += 1
            # plot graph #
            epi_total_reward_list.append(episode_total_reward)
            mean100=np.mean(epi_total_reward_list[-101:-1])
            mean_reward_list.append(mean100)
            plot_graph(mean_reward_list)
            print('episode %d timestep %d : threshold=%.5f total reward=%.2f'%(episode_num,time_step,threshold,episode_total_reward))
            episode_total_reward = 0
        if agent.buffer >= agent.batch_size:
            agent.update(0.99)
            update_times += 1

if __name__ == '__main__':
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    env=game.GameState()
    main()







        









