import torch
from torch import nn
import numpy as np
import PointNet_Common
class PointNet_Cls(nn.Module):
    def __init__(self,k=40):
        super.__init__(PointNet_Cls,self)
        self.TNet3 = PointNet_Common.T_Net3()
        self.FeatureNet = PointNet_Common.T_Net_Feature(k=64)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,64,1)
        self.conv3 = nn.Conv1d(64,128,1)
        self.conv4 = nn.Conv1d(128,1024,1)
        self.conv5 = nn.Conv1d(1024,512,1)
        self.conv6 = nn.Conv1d(512,256,1)
        self.conv7 = nn.Conv1d(256,self.k,1)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(0.7)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)

    def forward(self,x):
        batch_size = x.shape[0]
        # 一开始输入的x的shape (batch_size,nPoint,3)  nPoint是每个Batch的点的个数
        x = torch.transpose(x,1,2)
        # 现在的shape变成了(batch_size,channel=3,length= nPoint)
        rotateMatrix = self.TNet3(x)
        x = torch.transpose(x,1,2)   # 变回来(batch_size,nPoint,3) 以便进行Batched Matrix Multiply
        x = torch.matmul(x,rotateMatrix)
        x = torch.transpose(x,1,2)   # 重新变回(batch_size,channel=3,length) 进入Conv1d

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        featureMatrix = self.FeatureNet(x)
        x = torch.transpose(x,1,2)
        x = torch.matmul(x,featureMatrix)
        x = torch.transpose(x,1,2)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # 这里shape变为了： (batch_size,3,length)，需要做池化

        x = torch.max(x,dim=2,keepdim=True)[0]
        # x.shape (batch_size,1024,1)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.Dropout(x)
        x = self.conv7(x)
        # x.shape (batch_size,k,1)
        x = x.view(batch_size,self.k)
        return x,featureMatrix
