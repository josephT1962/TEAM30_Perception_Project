import torch
from torch import nn
import torchvision.models as models
from task.loss import DummyLoss
import numpy as np

'''
Extra Layer by Joe Team30


'''
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.branch = nn.Sequential(
                    nn.Linear(2048 ,512),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.2),
                    nn.Linear(512,3)
                )
        self.branch2 = nn.Sequential(
                nn.Linear(2048, 512),
                nn.Dropout(0.2),
                nn.Linear(512,3)
                )

    
    def forward(self, x):        
        return self.branch2(x)
'''
Change of activation functions , leaky relu
Change by Joe
'''
class JccNet(nn.Module):
    def __init__(self, **params):
        super(JccNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.avgpool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0) 
        #self.resnet.fc = nn.Linear(2048, 3) 
        self.resnet.fc = net()      
        self.dummyloss = DummyLoss()
    def forward(self, x):
        self.resnet.layer2[1].relu = nn.LeakyReLU(inplace=True)       
        self.resnet.layer2[2].relu = nn.LeakyReLU(inplace=True)   
        self.resnet.layer4[1].relu = nn.LeakyReLU(inplace=True)       
        self.resnet.layer4[2].relu = nn.LeakyReLU(inplace=True)       
        return self.resnet(x)

    def calc_loss(self, preds, gts):
        l = self.dummyloss(preds, gts)
        return [l]


