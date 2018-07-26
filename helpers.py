# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

"""A set of helper functions for dealing uniformly with tensors and
ndarrays."""

import numpy as np
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import ndimage

torch_tensor_types = tuple([
    torch.Tensor,
    torch.FloatTensor, torch.IntTensor, torch.LongTensor,
    torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.cuda.LongTensor
])

def asnd(x):
    """Convert torch/numpy to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Variable):
        x = x.data
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.IntTensor)):
        x = x.cpu()
    return x.numpy()

def as_nda(x, transpose_on_convert=None):
    """Turns any tensor into an ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    if isinstance(x, autograd.Variable):
        x = x.data
    if isinstance(x, torch_tensor_types):
        x = x.cpu().numpy()
        return np.ascontiguousarray(maybe_transpose(x, transpose_on_convert))
    raise ValueError("{}: can't convert to np.array".format(type(x)))

def astorch(x, single=True):
    """Convert torch/numpy to torch."""
    if isinstance(x, np.ndarray):
        if x.dtype == np.dtype("f"):
            return torch.FloatTensor(x)
        elif x.dtype == np.dtype("d"):
            if single:
                return torch.FloatTensor(x)
            else:
                return torch.DoubleTensor(x)
        elif x.dtype == np.dtype("i"):
            return torch.IntTensor(x)
        else:
            error("unknown np.dtype")
    return x

def as_torch(x, transpose_on_convert=None, single=True):
    """Converts any kind of tensor/array into a torch tensor."""
    if isinstance(x, Variable):
        return x.data
    if isinstance(x, torch_tensor_types):
        return x
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = maybe_transpose(x, transpose_on_convert)
        if x.dtype == np.dtype("f"):
            return torch.FloatTensor(x)
        elif x.dtype == np.dtype("d"):
            if single:
                return torch.FloatTensor(x)
            else:
                return torch.DoubleTensor(x)
        elif x.dtype in [np.dtype("i"), np.dtype("int64")]:
            return torch.LongTensor(x)
        else:
            raise ValueError("{} {}: unknown dtype".format(x, x.dtype))
    raise ValueError("{} {}: unknown type".format(x, type(x)))

def is_tensor(x):
    if isinstance(x, Variable):
        x = x.data
    return isinstance(x, torch_tensor_types)

def rank(x):
    """Return the rank of the ndarray or tensor."""
    if isinstance(x, np.ndarray):
        return x.ndim
    else:
        return x.dim()

def size(x, i):
    """Return the size of dimension i."""
    if isinstance(x, np.ndarray):
        return x.shape[i]
    else:
        return x.size(i)

def shp(x):
    """Returns the shape of a tensor or ndarray as a tuple."""
    if isinstance(x, Variable):
        return tuple(x.data.size())
    elif isinstance(x, np.ndarray):
        return tuple(x.shape)
    elif isinstance(x, torch_tensor_types):
        return tuple(x.size())
    else:
        raise ValueError("{}: unknown type".format(type(x)))

def novar(x):
    """Turns a variable into a tensor; does nothing for a tensor."""
    if isinstance(x, Variable):
        return x.data
    return x

def maybe_transpose(x, axes):
    if axes is None: return x
    return x.transpose(axes)

def typeas(x, y):
    """Make x the same type as y, for numpy, torch, torch.cuda."""
    assert not isinstance(x, Variable)
    if isinstance(y, Variable):
        y = y.data
    if isinstance(y, np.ndarray):
        return asnd(x)
    if isinstance(x, np.ndarray):
        if isinstance(y, (torch.FloatTensor, torch.cuda.FloatTensor)):
            x = torch.FloatTensor(x)
        else:
            x = torch.DoubleTensor(x)
    return x.type_as(y)

def sequence_is_normalized(x, d, eps=1e-3):
    """Check whether a batch of sequences BDL is normalized in d."""
    if isinstance(x, Variable):
        x = x.data
    assert x.dim() == 3
    marginal = x.sum(d)
    return (marginal - 1.0).abs().lt(eps).all()

def bhwd2bdhw(images, depth1=False):
    images = as_torch(images)
    if depth1:
        assert len(shp(images)) == 3, shp(images)
        images = images.unsqueeze(3)
    assert len(shp(images)) == 4, shp(images)
    return images.permute(0, 3, 1, 2)

def bdhw2bhwd(images, depth1=False):
    images = as_torch(images)
    assert len(shp(images)) == 4, shp(images)
    images = images.permute(0, 2, 3, 1)
    if depth1:
        assert images.size(3) == 1
        images = images.index_select(3, 0)
    return images

def reorder(batch, inp, out):
    """Reorder the dimensions of the batch from inp to out order.

    E.g. BHWD -> BDHW.
    """
    if inp is None: return batch
    if out is None: return batch
    assert isinstance(inp, str)
    assert isinstance(out, str)
    assert len(inp) == len(out), (inp, out)
    assert rank(batch) == len(inp), (rank(batch), inp)
    result = [inp.find(c) for c in out]
    # print ">>>>>>>>>>>>>>>> reorder", result
    for x in result: assert x >= 0, result
    if is_tensor(batch):
        return batch.permute(*result)
    elif isinstance(batch, np.ndarray):
        return batch.transpose(*result)

def assign(dest, src, transpose_on_convert=None):
    """Resizes the destination and copies the source."""
    src = as_torch(src, transpose_on_convert)
    if isinstance(dest, Variable):
        dest.data.resize_(*shp(src)).copy_(src)
    elif isinstance(dest, torch.Tensor):
        dest.resize_(*shp(src)).copy_(src)
    else:
        raise ValueError("{}: unknown type".format(type(dest)))

def one_sequence_softmax(x):
    """Compute softmax over a sequence; shape is (l, d)"""
    y = asnd(x)
    assert y.ndim==2, "%s: input should be (length, depth)" % y.shape
    l, d = y.shape
    y = np.amax(y, axis=1)[:, np.newaxis] -y
    y = np.clip(y, -80, 80)
    y = np.exp(y)
    y = y / np.sum(y, axis=1)[:, np.newaxis]
    return typeas(y, x)

def sequence_softmax(x):
    """Compute sotmax over a batch of sequences; shape is (b, l, d)."""
    y = asnd(x)
    assert y.ndim==3, "%s: input should be (batch, length, depth)" % y.shape
    for i in range(len(y)):
        y[i] = one_sequence_softmax(y[i])
    return typeas(y, x)

def ctc_align(prob, target):
    """Perform CTC alignment on torch sequence batches (using ocrolstm)"""
    import cctc
    prob_ = prob.cpu()
    target = target.cpu()
    b, l, d = prob.size()
    bt, lt, dt = target.size()
    assert bt==b, (bt, b)
    assert dt==d, (dt, d)
    assert sequence_is_normalized(prob, 2), prob
    assert sequence_is_normalized(target, 2), target
    result = torch.rand(1)
    cctc.ctc_align_targets_batch(result, prob_, target)
    return typeas(result, prob)

def ctc_loss(logits, target):
    """A CTC loss function for BLD sequence training."""
    assert logits.is_contiguous()
    assert target.is_contiguous()
    probs = sequence_softmax(logits)
    aligned = ctc_align(probs, target)
    assert aligned.size()==probs.size(), (aligned.size(), probs.size())
    deltas = aligned - probs
    logits.backward(deltas.contiguous())
    return deltas, aligned

class LearningRateSchedule(object):
    def __init__(self, schedule):
        if ":" in schedule:
            self.learning_rates = [[float(y) for y in x.split(",")] for x in schedule.split(":")]
            assert self.learning_rates[0][0] == 0
        else:
            lr0 = float(schedule)
            self.learning_rates = [[0, lr0]]
    def __call__(self, count):
        _, lr = self.learning_rates[0]
        for n, l in self.learning_rates:
            if count < n: break
            lr = l
        return lr

