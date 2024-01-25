import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


#=======================================================================================================================
# WEIGHTS INITS
#=======================================================================================================================
def xavier_init(m):
    s =  np.sqrt( 2. / (m.in_features + m.out_features) )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

#=======================================================================================================================
def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)

#=======================================================================================================================

#=======================================================================================================================
# ACTIVATIONS
#=======================================================================================================================
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat( F.relu(x), F.relu(-x), 1 )

#=======================================================================================================================
# LAYERS
#=======================================================================================================================
class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=False, gated=False, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.gated = gated
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)
        if self.gated:
            self.sigmoid = nn.Sigmoid()
            self.g = nn.Linear(int(input_size), int(output_size))

    def forward(self, x):
        h = self.linear(x)
        if self.gated:
            g = self.sigmoid( self.g(x) )
            h = h * g
        elif self.activation is not None:
            h = self.activation( h )

        return h

#=======================================================================================================================
class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

#=======================================================================================================================
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None, bias=False):
        super(Conv2d, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, x):
        h = self.conv(x)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)

        return out

# =======================================================================================================================
