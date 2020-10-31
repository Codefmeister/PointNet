import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class T_Net3(nn.Module):
    def __init__(self):
        super(T_Net3,self)
        # input data shape: (batch_size, channel, length)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,9)    # softMax

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
    def forward(self,x):
        # input shape: (batch_size, channel, length)
        batch_size = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x,2,True)[0]    # dim=2,作用范围是length，也就是将很多点的特征做了池化，池化为一个点（length实际上就是一个batch里点的数目）
        # 所以到这里，x的shape应该是(batch_size,1024,1)
        x = x.view(-1,1024)     # 每个batch生成一个3x3，输入fc全连接层前，要先将feature维摆在最后
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.relu(self.fc3(x))
        # 初始化是一个identity矩阵   这里有点懵逼，网上实现的使用了Variable，将tensor传入Variable进行相加。
        # 但我在Jupyter中测试感觉两者并无区别，先用Tensor写着。
        identity = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1],dtype=np.float32)).view(1,9).repeat(batch_size,1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(batch_size, 3, 3)
        return x

class T_Net_Feature(nn.Module):
    def __init__(self,k=64):
        super.__init__(T_Net_Feature, self)
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, k*k)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self,x):
        # 输入的x shape: (batch_size,channel,length)
        batch_size = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # 此时shape应该是 (batch_size,1024,length) 下一步进行池化
        x = torch.max(x,dim=2,keepdim=True)[0]
        # 池化后的shape (batch_size, 1024, 1)
        # 进入全连接层之前需要先改变shape
        x = x.view(batch_size,1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.from_numpy(np.eye(self.k),dtype = np.float32).view(1,self.k*self.k).repeat((batch_size,1))
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(batch_size,self.k,self.k)
        return x