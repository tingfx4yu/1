import math
import random
from collections import namedtuple, deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from IPython.display import clear_output
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args,**kwargs).cuda if USE_CUDA else autograd.Variable(*args,**kwargs)

class StochasticMDP:
    def __init__(self):
        self.end = False
        self.current_state = 2
        self.num_actions = 2
        self.num_states = 6
        self.P_right = 0.5

    def reset(self):
        self.end = False
        self.current_state = 2
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self,action):
        if self.current_state != 1:
            if action == 1:
                if random.random() < self.P_right and self.current_state < self.num_states:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            if action == 0:
                self.current_state -= 1

            if self.current_state == self.num_states:
                self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1

        if self.current_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state 1.00/100.00, True, {}
        else:
            return state, 0.0, False, {}
        
class ReplayBuffer(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state,0)
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer,batch_size))
        return np.concatenat(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Net,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs,256),
            nn.ReLU(),
            nn.Linear(256,num_outputs)
        )
    
    def forward(self,x):
        return self.layers(x)

    def act(self,state,epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.forward(Variable(state)).max(1)[1]
            return action.data[0]
        else:
            return random.randrange(num_actions)

env = StochasticMDP()
num_goals = env.num_states
num_actions = env.num_actions

model = Net(2*num_goals,num_actions)
target_model = Net(2*num_goals,num_actions)

meta_model = Net(num_goals,num_goals)
target_meta_model = Net(num_goals,num_goals)

if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    meta_model = meta_model.cuda()
    target_meta_model = target_meta_model.cuda()

optimizer = optim.Adam(model.parameters())
meta_optimizer = optim.Adam(meta_model.parameters())

replay_buffer = ReplayBuffer(10000)
meta_replay_buffer = ReplayBuffer(10000)

def to_onehot(x):
    oh = np.zeros(6)
    oh[x-1] = 1
    return oh

def update(model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(state))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(next_state))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_value = model(state)
    q_value = q_value.gather(1,action.unsqueeze(1)).squeeze(1)

    next_q_value = model(next_state).max(1)[0]
    expect_q_value = model(next_state).max(1)[0]

    loss = (q_value - Variable(expect_q_value)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500



