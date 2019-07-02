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

# if gpu is to be used #

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

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

class dqn_net(nn.Module):
    def __init__(self,ACTION_NUM):
        super(dqn_net,self).__init__()
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
    def select_action(self,input):
        '''
        parameters
        ----------
        input : {Tensor} of shape torch.Size([4,84,84])

        Return
        ------
        action_button , action_onehot : {int} , {Tensor}
        '''
        input=Variable(input.unsqueeze(0))
        output=self.forward(input)
        action_index=output.data.max(1)[1][0]
        action_button=action_index
        action_onehot=Tensor([0]*self.action_num)
        action_onehot[action_index]=1
        return action_button,action_onehot
    def update(self,samples,loss_func,optim_func,learn_rate,target_net,BATCH_SIZE,GAMMA):
        '''update the value network one step

        Parameters
        ----------
        samples: {namedtuple}
            Transition(obs4=(o1,o2,...),act=(a1,a2,...),
            next_ob=(no1,no2,...),reward=(r1,r2,...),done=(d1,d2,...))
        loss: string
            the loss function of the network
            e.g. 'nn.MSELoss'
        optim: string
            the optimization function of the network
            e.g. 'optim.SGD'
        learn_rate: float
            the learing rate of the optimizer

        Functions
        ---------
            update the network one step
        '''
        obs4_batch=Variable(torch.cat(samples.obs4)) # ([BATCH,4,84,84]) {Variable}
        next_obs4_batch=Variable(torch.cat(samples.next_obs4)) # ([BATCH,4,84,84]) {Variable}
        action_batch=Variable(torch.cat(samples.act)) # ([BATCH,6]) {Variable}
        done_batch=samples.done # {tuple} of bool,len=BATCH
        reward_batch=torch.cat(samples.reward) # ([BATCH,1]) {FloatTensor}
        ### compute the target Q(s,a) value ###
        value_batch=target_net(next_obs4_batch) # ([B,6])
        target=Variable(torch.zeros(BATCH_SIZE).type(Tensor)) # ([B])
        for i in range(BATCH_SIZE):
            if done_batch[i]==False:
                target[i]=reward_batch[i]+GAMMA*torch.max(value_batch.data[i])
            elif done_batch[i]==True:
                target[i]=reward_batch[i]
        ### compute the current net output value ###
        action_idx_batch=action_batch.max(dim=1)[1].unsqueeze(1) # {Variable} ([B,1])
        output=self.forward(obs4_batch) # {Variable} ([B,6])
        q_action=output.gather(dim=1,index=action_idx_batch) # {Variable contain FloatTensor} ([B,1])
        net_output=q_action.squeeze(1) # ([B])

        #criterion=loss_func()
        optimizer=torch.optim.RMSprop(self.parameters(),lr=0.00025,alpha=0.95,eps=0.01)
        #loss=criterion(net_output,target)
        ########
        criterion=nn.SmoothL1Loss()
        loss=criterion(input=net_output,target=target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #############
        '''
        print('before clip\n####################')
        for param in self.parameters():
            print(param.size(),'max:',param.data.max(),'min:',param.data.min(),'\n')
        '''
        torch.nn.utils.clip_grad_norm(parameters=self.parameters(), max_norm=1)
        '''
        print('after clip\n####################')
        for param in self.parameters():
            print(param.size(), 'max:', param.data.max(), 'min:', param.data.min(),'\n')
        '''
        optimizer.step()


class replay_memory:
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
        self.Transition=namedtuple('Transition',
                                   ['obs4','act','next_obs4','reward','done'])
    def __len__(self):
        return len(self.memory)
    def add(self,*args):
        '''Add a transition to replay memory
        Parameters
        ----------
        e.g. repay_memory.add(obs4,action,next_obs4,reward,done)
        obs4: {Tensor} of shape torch.Size([4,84,84])
        act: {Tensor} of shape torch.Size([action_num])
        next_obs4: {Tensor} of shape torch.Size([4,84,84])
        reward: {int}
        done: {bool} the next station is the terminal station or not

        Function
        --------
        the replay_memory will save the latest samples
        '''
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position]=self.Transition(*args)
        self.position=(self.position+1)%self.capacity
    def sample(self,batch_size):
        '''Sample a batch from replay memory
        Parameters
        ----------
        batch_size: int
            How many trasitions you want

        Returns
        -------
        obs_batch: {Tensor} of shape torch.Size([BATCH_SIZE,4,84,84])
            batch of observations

        act_batch: {Tensor} of shape torch.Size([BATCH_SIZE,action_num])
            batch of actions executed w.r.t observations in obs_batch

        nob_batch: {Tensor} of shape torch.Size([BATCH_SIZE,4,84,84])
            batch of next observations w.r.t obs_batch and act_batch

        rew_batch: {ndarray} of shape
            batch of reward received w.r.t obs_batch and act_batch
        '''
        batch = random.sample(self.memory, batch_size)
        batch_zip=self.Transition(*zip(*batch))
        return batch_zip

def learn(env,
        MAX_EPISODE,
        EPS_START,
        EPS_END,
        EXPLO_FRAC,
        ACTION_NUM,
        REPLAY_MEMORY_CAPACITY,
        BATCH_SIZE,
        LOSS_FUNCTION,
        OPTIM_METHOD,
        LEARNING_RATE,
        GAMMA,
        NET_COPY_STEP,
        OBSERVE,
        TRAIN_FREQ,
        FRAME_PER_ACTION,
        MODELPATH,
        DATAPATH
        ):
    ### initialization ###
    #os.mkdir(DATAPATH)
    action_meaning = [
        np.array([1,0]),
        np.array([0,1])
    ] # 游戏有惯性，停止采取运动后，之后的1、2、3帧还会运动，第4帧停下
    action_space=[]
    for i in range(ACTION_NUM):
        action_onehot=Tensor([0]*ACTION_NUM)
        action_onehot[i]=1
        action_space.append((i,action_onehot))
    value_net = dqn_net(ACTION_NUM)
    target_net=dqn_net(ACTION_NUM)
    if torch.cuda.is_available():
       value_net.cuda()
       target_net.cuda()
    if os.path.exists(MODELPATH):
        value_net.load_state_dict(torch.load(PATH))
    buffer=replay_memory(REPLAY_MEMORY_CAPACITY)
    action0=action_meaning[action_space[0][0]]
    obs,_,_=env.frame_step(action0) # {ndarray}
    obs=ob_process(obs)
    obs4=torch.cat(([obs,obs,obs,obs]),dim=0) # {Tensor} of shape torch.Size([4,84,84])
    '''
    obs4=np.resize(obs,(4,84,84)).astype('float64')
    obs4=Tensor(obs4)
    '''
    episode_total_reward = 0
    epi_total_reward_list=[]
    mean_reward_list=[]
    # counters #
    time_step=0
    update_times=0
    episode_num=0

    while episode_num < MAX_EPISODE:
        ### choose an action with epsilon-greedy ###
        #env.render()
        prob = random.random()

        a = (EPS_START - EPS_END)
        b = (EXPLO_FRAC - max(0, time_step - OBSERVE))
        threshold = EPS_END + max(0, a * b / EXPLO_FRAC)
        if time_step % FRAME_PER_ACTION ==0:
            
            if prob <= threshold:
                action_index = np.random.randint(ACTION_NUM)
                action_button = action_space[action_index][0] # {int}
                action_onehot = action_space[action_index][1] # {Tensor}
            else:
                action_button, action_onehot = value_net.select_action(obs4)
            
            #action_button, action_onehot = value_net.select_action(obs4)
        else:
            action_button=1
        ### do one step ###
        action=action_meaning[action_button]
        #print(action)
        obs_next, reward, done = env.frame_step(action)
        obs_next = ob_process(obs_next)
        obs4_next = torch.cat(([obs4[1:, :, :],obs_next]),dim=0)
        '''
        obs4_next=np.resize(obs_next,(4,84,84)).astype('float64')
        obs4_next=Tensor(obs4_next)
        '''
        buffer.add(obs4.unsqueeze(0), action_onehot.unsqueeze(0), obs4_next.unsqueeze(0), Tensor([reward]).unsqueeze(0), done)
        episode_total_reward+=reward
        '''the transition added to buffer
        obs4: {ndarray} size (4,84,84)
        action: {list} size action_num
        obs_next: {ndarray} size (84,84)
        reward: {int}
        done: {bool}
        '''
        ### go to the next state ###
        obs4 = obs4_next
        if done == False:

            time_step += 1
        elif done == True:
            #env.reset()
            #obs = ob_process(obs)
            #obs4 = torch.cat(([obs, obs, obs, obs]), dim=0)

            episode_num += 1
            # plot graph #
            epi_total_reward_list.append(episode_total_reward)
            mean100=np.mean(epi_total_reward_list[-101:-1])
            mean_reward_list.append(mean100)
            plot_graph(mean_reward_list)
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
            value_net.update(samples=batch_transition, loss_func=LOSS_FUNCTION,
                             optim_func=OPTIM_METHOD, learn_rate=LEARNING_RATE,
                             target_net=target_net, BATCH_SIZE=BATCH_SIZE,
                             GAMMA=GAMMA)
            update_times += 1

            ### copy value net parameters to target net ###
            if update_times % NET_COPY_STEP == 0:
                target_net.load_state_dict(value_net.state_dict())

    torch.save(value_net.state_dict(),MODELPATH)


if __name__=='__main__':
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    env=game.GameState()
    learn(env=env,
        MAX_EPISODE=200000,
        EPS_START=1.0,
        EPS_END=0.1,
        EXPLO_FRAC=170000,
        ACTION_NUM=2,
        REPLAY_MEMORY_CAPACITY=50000,#1000000,
        BATCH_SIZE=32,
        LOSS_FUNCTION=nn.MSELoss,
        OPTIM_METHOD=optim.Adam,
        LEARNING_RATE=1e-6,
        GAMMA=0.99,
        NET_COPY_STEP=100,#10000,
        OBSERVE=100,#50000,
        TRAIN_FREQ=1,
        FRAME_PER_ACTION=1,
        MODELPATH='./data_save/dqn/param.pt',
        DATAPATH='./data_save/dqn/'
        )