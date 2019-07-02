import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor
'''
image = Variable(torch.randn(1,3,3))
state = torch.cat(tuple(image for _ in range(4)))[None,:,:,:]
#print('image = ',image)
#print(state.shape)
#print('state = ',state)
x = torch.FloatTensor([[1,2,5,3]])
o = [x,x,x,x,x]
y = x.unsqueeze(1)
#print('x_max_index =',x.max(0)[1].unsqueeze(1))
#print('y=',y)
z = torch.cat((x,x))
#print(z)
#print(z.max(1)[1])

a_s = []
for i in range(8):
    a_o = Tensor([0]*8)
    a_o[i] = 1
    a_s.append((i,a_o))
print('a_s = ',a_s)
print('a_s[0][0] =',a_s[0][0])

print(o)
print(torch.stack(o))
'''
class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.size())

if __name__ == '__main__':
    net =DQN()
    data_input = Variable(torch.randn([1,4,128,128]))
    print(data_input.size())
    net(data_input)
