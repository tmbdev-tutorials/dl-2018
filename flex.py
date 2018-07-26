# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.legacy import nn as legnn
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.legacy import nn as legnn
import layers

class Flex(nn.Module):
    def __init__(self, creator):
        super(Flex, self).__init__()
        self.creator = creator
        self.layer = None
	self.dummy = nn.Parameter(torch.zeros(1))
    def forward(self, *args):
        if self.layer is None:
            self.layer = self.creator(*args)
            self.layer.to(self.dummy.device)
        return self.layer.forward(*args)
    def __repr__(self):
        return "Flex:"+repr(self.layer)
    def __str__(self):
        return "Flex:"+str(self.layer)

def Linear(*args, **kw):
    def creator(x):
        assert x.ndimension()==2
        return nn.Linear(x.size(1), *args, **kw)
    return Flex(creator)


def Conv1d(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        d = x.size(1)
        return nn.Conv1d(x.size(1), *args, **kw)
    return Flex(creator)


def Conv2d(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        return nn.Conv2d(x.size(1), *args, **kw)
    return Flex(creator)


def Conv3d(*args, **kw):
    def creator(x):
        assert x.ndimension()==5
        return nn.Conv3d(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm1(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        return layers.LSTM1(x.size(1), *args, **kw)
    return Flex(creator)


def LSTM1to0(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        return layers.Lstm1to0(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm2(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        return layers.LSTM2(x.size(1), *args, **kw)
    return Flex(creator)


def Lstm2to1(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        return layers.LSTM2to1(x.size(1), *args, **kw)
    return Flex(creator)

def BatchNorm1d(*args, **kw):
    def creator(x):
        assert x.ndimension()==3
        return nn.BatchNorm1d(x.size(1), *args, **kw)
    return Flex(creator)

def BatchNorm2d(*args, **kw):
    def creator(x):
        assert x.ndimension()==4
        return nn.BatchNorm2d(x.size(1), *args, **kw)
    return Flex(creator)

def BatchNorm3d(*args, **kw):
    def creator(x):
        assert x.ndimension()==5
        return nn.BatchNorm3d(x.size(1), *args, **kw)
    return Flex(creator)

def replace_modules(model, f):
    for key in model._modules.keys():
        sub = model._modules[key]
        replacement = f(sub)
        if replacement is not None:
            model._modules[key] = replacement
        else:
            replace_modules(sub, f)

def flex_replacer(module):
    if isinstance(module, Flex):
        return module.layer
    else:
        return None

def flex_freeze(model):
    replace_modules(model, flex_replacer)

def delete_modules(model, f):
    for key in model._modules.keys():
        if f(model._modules[key]):
            del model._modules[key]

