import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym
from torch import optim

#Hyper-parameter
MEMORY_CAPACITY = 2000
LR = 0.01
GAMMA = 0.9
env = gym.make('MountainCar-v0')
env = env.unwrapped
NUM_STATE = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n
device = torch.device(“cuda:0” if torch.cuda.is_available() else “cpu”)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(NUM_STATE,30)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(30,NUM_ACTIONS)
        self.fc2.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class DQN():
    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()
        self.memory = np.zeros((MEMORY_CAPACITY,NUM_STATE*2+2))
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss = nn.MSELoss()

        self.fig, self.ax = plt.subplots()

    def store(self,state,action,reward,next_state):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state,[action],[reward],next_state))
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state),0)
        if np.random.randn() <= 0.9:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value,1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0,NUM_ACTIONS)
        return action

    def plot(self,ax,x):
        ax.cla()
        ax.set_xlabel('episode')
        ax.set_ylabel('total reward')
        ax.plot(x,'b-')
        plt.pause(0.0001)

    def learn(self):
        if self.learn_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY,32)
        batch_memory = self.memory[sample_index,:]
        batch_state = torch.FloatTensor(batch_memory[:,:NUM_STATE]).to(device)
        batch_action = torch.LongTensor(batch_memory[:,NUM_STATE:NUM_STATE+1].astype(int)).to(device)
        batch_reward = torch.FloatTensor(batch_memory[:,NUM_STATE+1:NUM_STATE+2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATE:]).to(device)

        q_eval = self.eval_net(batch_state).gather(1,batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA*q_next.max(1)[0].view(32,1)

        loss = self.loss(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    net = DQN()
    print('The DQN is collecting experience.....')
    step_counter_list = []
    for episode in range(400):
        state = env.reset()
        step_counter = 0
        while True:
            step_counter += 1
            env.render()
            action = net.choose(state)
            next_state,reward,done,info = env.step(action)
            reward = reward * 100 if reward >0 else reward * 5
            net.store(state,action,reward,next_state)

            if net.memory_counter > MEMORY_CAPACITY:
                net.learn()
                if done:
                    print('episode {}, the reward is {}'.format(episode,round(reward,3)))
            if done:
                step_counter_list.append(step_counter)
                net.plot(net.ax,step_counter_list)
                break
            state = next_state

if __name__ == '__main__':
    main()

