import torch
import torch.nn as nn
from RainforestDataset import get_classes_list

torch.manual_seed(0)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        self.fully_conv1 = pretrained_net1
        self.fully_conv2 = pretrained_net2
        self.fully_conv1.fc = nn.Identity()
        self.fully_conv2.fc = nn.Identity()

        self.linear = nn.Linear(in_features=512*2, out_features=num_classes)
        
        self.sig = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        out1 = self.fully_conv1(inputs1)
        out2 = self.fully_conv2(inputs2)
        out_cat = torch.cat((out1, out2), dim=1)
        out_cat = out_cat.view(out_cat.size(0), -1)
        out = self.linear(out_cat)
        return self.sig(out)

class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init=None):
        super(SingleNetwork, self).__init__()

        _, num_classes = get_classes_list()


        if weight_init is not None:

            new_in_channels = 4
            layer = pretrained_net.conv1

            new_layer = nn.Conv2d(in_channels=new_in_channels, 
                  out_channels=layer.out_channels, 
                  kernel_size=layer.kernel_size, 
                  stride=layer.stride, 
                  padding=layer.padding,
                  bias=layer.bias)

            copy_weights = layer.weight.clone()

            if weight_init == "kaiminghe":
                w = layer.weight.clone()
                nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
                w = w[:, 0:1, :, :]
                new_weight = torch.cat((copy_weights, w), dim=1)
                new_layer.weight = torch.nn.Parameter(new_weight)
                
            pretrained_net.conv1 = new_layer

        pretrained_net.fc = nn.Sequential(nn.Linear(in_features=pretrained_net.fc.in_features, out_features=num_classes))

        self.net = pretrained_net
        self.sigm = nn.Sigmoid()

    def forward(self, inputs):
        return self.sigm(self.net(inputs))