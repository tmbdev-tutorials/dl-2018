# copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

"""A set of "trainers", classes that wrap around Torch models
and provide methods for training and evaluation."""

import time
import types
import platform
import numpy as np
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable, Function
import torch.nn.functional as F
from scipy import ndimage
import helpers as dlh

def add_log(log, logname, **kw):
    entry = dict(kw, __log__=logname, __at__=time.time(), __node__=platform.node())
    log.append(entry)

def get_log(log, logname, **kw):
    records = [x for x in log if x.get("__log__")==logname]
    return records

def update_display():
    from matplotlib import pyplot
    from IPython import display
    display.clear_output(wait=True)
    display.display(pyplot.gcf())

class Weighted(Function):
    def forward(self, x, weights):
        self.saved_for_backward = [weights]
        return x
    def backward(self, grad_output):
        weights, = self.saved_for_backward
        grad_input = weights * grad_output
        return grad_input


class BasicTrainer(object):
    """Trainers take care of bookkeeping for training models.

    The basic method is `train_batch(inputs, targets)`. It catches errors
    during forward propagation and reports the model and input shapes
    (shape mismatches are the most common source of errors.

    Trainers are just a temporary tool that's wrapped around a model
    for training purposes, so you can create, use, and discard them
    as convenient.
    """

    def __init__(self, model, use_cuda=True,
                 fields = ("input", "output"),
                 input_axes = None,
                 output_axes = None):
        self.use_cuda = use_cuda
        self.model = self._cuda(model)
        self.init_loss()
        self.input_name, self.output_name = fields
        self.no_display = False
        self.current_lr = None
	self.optimizer = None
        self.weighted = Weighted()
        self.ntrain = 0
        self.log = []

    def _cuda(self, x):
        """Convert object to CUDA if use_cuda==True."""
        if self.use_cuda:
            return x.cuda()
        else:
            return x.cpu()

    def set_training(self, mode=True):
        """Set training or prediction mode."""
        if mode:
            if not self.model.training:
                self.model.train()
            self.cuinput = autograd.Variable(
                torch.randn(1, 1, 100, 100).cuda())
            self.cutarget = autograd.Variable(torch.randn(1, 11).cuda())
        else:
            if self.model.training:
                self.model.eval()
            self.cuinput = autograd.Variable(torch.randn(1, 1, 100, 100).cuda(),
                                             volatile=True)
            self.cutarget = autograd.Variable(torch.randn(1, 11).cuda(),
                                              volatile=True)

    def set_lr(self, lr, momentum=0.9, weight_decay=0.0):
        """Set the optimizer to SGD with the given parameters."""
        self.current_lr = lr
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

    def get_outputs(self):
        """Performs any necessary transformations on the output tensor.
        """
        return dlh.novar(self.cuoutput).cpu()

    def set_inputs(self, batch):
        """Sets the cuinput variable from the input data.
        """
        assert isinstance(batch, torch.Tensor)
        dlh.assign(self.cuinput, batch)

    def set_targets(self, targets, weights=None):
        """Sets the cutarget variable from the given tensor.
        """
        dlh.assign(self.cutarget, targets, False)
        assert self.cuoutput.size() == self.cutargets.size()
        if weights is not None:
            dlh.assign(self.cuweights, weights, False)
            assert self.cuoutput.size() == self.cuweights.size()
        else:
            self.cuweights = None

    def init_loss(self, loss=nn.MSELoss()):
        self.criterion = self._cuda(loss)

    def compute_loss(self, targets, weights=None):
        self.set_targets(targets, weights=weights)
        return self.criterion(self.cuoutput, self.cutarget)

    def forward(self):
        try:
            self.cuoutput = self.model(self.cuinput)
        except RuntimeError, err:
            print "runtime error in forward step:"
            print "input", self.cuinput.size()
            raise err

    def train_batch(self, inputs, targets, weights=None, update=True, logname="train"):
        if update:
            self.set_training(True)
            self.optimizer.zero_grad()
        else:
            self.set_training(False)
        self.set_inputs(inputs)
        self.forward()
        if weights is not None:
            self.cuweights = autograd.Variable(torch.randn(1, 1).cuda())
            dlh.assign(self.cuweights, weights, False)
            self.cuoutput = self.weighted(self.cuoutput, self.cuweights)
        culoss = self.compute_loss(targets, weights=weights)
        if update:
            culoss.backward()
            self.optimizer.step()
        ploss = dlh.novar(culoss)[0]
        self.ntrain += dlh.size(inputs, 0)
        add_log(self.log, logname, loss=ploss, ntrain=self.ntrain, lr=self.current_lr)
        return self.get_outputs(), ploss

    def eval_batch(self, inputs, targets):
        return self.train_batch(inputs, targets, update=False, logname="eval")

    def predict_batch(self, inputs):
        self.set_training(False)
        self.set_inputs(inputs)
        self.forward()
        return self.get_outputs()

    def loss_curve(self, logname):
        records = get_log(self.log, logname)
        records = [(x["ntrain"], x["loss"]) for x in records]
        records = sorted(records)
        if len(records)==0:
            return [], []
        else:
            return zip(*records)

    def plot_loss(self, every=100, smooth=1e-2, yscale=None):
        if self.no_display: return
        # we import these locally to avoid dependence on display
        # functions for training
        import matplotlib as mpl
        from matplotlib import pyplot
        from scipy.ndimage import filters
        x, y = self.loss_curve("train")
        pyplot.plot(x, y)
        x, y = self.loss_curve("test")
        pyplot.plot(x, y)

    def display_loss(self, *args, **kw):
        pyplot.clf()
        self.plot_loss(*args, **kw)
        update_display()

    def set_sample_fields(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name

    def train_for(self, training, training_size=1e99):
        if isinstance(training, types.FunctionType):
            training = training()
        count = 0
        losses = []
        for batch in training:
            if count >= training_size: break
            input_tensor = batch[self.input_name]
            output_tensor = batch[self.output_name]
            _, loss = self.train_batch(input_tensor, output_tensor)
            count += len(input_tensor)
            losses.append(loss)
        loss = np.mean(losses)
        return loss, count

    def eval_for(self, testset, testset_size=1e99):
        if isinstance(testset, types.FunctionType):
            testset = testset()
        count = 0
        losses = []
        for batch in testset:
            if count >= testset_size: break
            input_tensor = batch[self.input_name]
            output_tensor = batch[self.output_name]
            _, loss = self.eval_batch(input_tensor, output_tensor)
            count += len(input_tensor)
            losses.append(loss)
        loss = np.mean(losses)
        return loss, count

class ImageClassifierTrainer(BasicTrainer):
    def __init__(self, *args, **kw):
        BasicTrainer.__init__(self, *args, **kw)

    def set_inputs(self, images, depth1=False):
        dlh.assign(self.cuinput, images, transpose_on_convert=(0, 3, 1, 2))

    def set_targets(self, targets, weights=None):
        assert weights is None, "weights not implemented"
	if isinstance(targets, list):
	    targets = np.array(targets)
        if dlh.rank(targets) == 1:
            targets = dlh.as_torch(targets)
            targets = targets.unsqueeze(1)
            b, c = dlh.shp(self.cuoutput)
            onehot = torch.zeros(b, c)
            onehot.scatter_(1, targets, 1)
            dlh.assign(self.cutarget, onehot)
        else:
            assert dlh.shp(targets) == dlh.shp(self.cuoutput)
            dlh.assign(self.cutarget, targets)


def zoom_like(batch, target_shape, order=0):
    assert isinstance(batch, np.ndarray)
    scales = [r * 1.0 / b for r, b in zip(target_shape, batch.shape)]
    result = np.zeros(target_shape)
    ndimage.zoom(batch, scales, order=order, output=result)
    return result

def pixels_to_batch(x):
    b, d, h, w = x.size()
    return x.permute(0, 2, 3, 1).contiguous().view(b*h*w, d)

class Image2ImageTrainer(BasicTrainer):
    """Train image to image models."""
    def __init__(self, *args, **kw):
        BasicTrainer.__init__(self, *args, **kw)

    def compute_loss(self, targets, weights=None):
        self.set_targets(targets, weights=weights)
        return self.criterion(pixels_to_batch(self.cuoutput),
                              pixels_to_batch(self.cutarget))

    def set_inputs(self, images):
        dlh.assign(self.cuinput, images, (0, 3, 1, 2))

    def get_outputs(self):
        return dlh.as_nda(self.cuoutput, (0, 2, 3, 1))

    def set_targets(self, targets, weights=None):
        b, d, h, w = tuple(self.cuoutput.size())
        targets = dlh.as_nda(targets, (0, 2, 3, 1))
        targets = zoom_like(targets, (b, h, w, d))
        dlh.assign(self.cutarget, targets, (0, 3, 1, 2))
        assert self.cutarget.size() == self.cuoutput.size()
        if weights is not None:
            weights = dlh.as_nda(weights, (0, 2, 3, 1))
            weights = zoom_like(weights, (b, h, w, d))
            dlh.assign(self.cuweights, weights, (0, 3, 1, 2))

def ctc_align(prob, target):
    """Perform CTC alignment on torch sequence batches (using ocrolstm).

    Inputs are in BDL format.
    """
    import cctc
    assert dlh.sequence_is_normalized(prob), prob
    assert dlh.sequence_is_normalized(target), target
    # inputs are BDL
    prob_ = dlh.novar(prob).permute(0, 2, 1).cpu().contiguous()
    target_ = dlh.novar(target).permute(0, 2, 1).cpu().contiguous()
    # prob_ and target_ are both BLD now
    assert prob_.size(0) == target_.size(0), (prob_.size(), target_.size())
    assert prob_.size(2) == target_.size(2), (prob_.size(), target_.size())
    assert prob_.size(1) >= target_.size(1), (prob_.size(), target_.size())
    result = torch.rand(1)
    cctc.ctc_align_targets_batch(result, prob_, target_)
    return dlh.typeas(result.permute(0, 2, 1).contiguous(), prob)

def sequence_softmax(seq):
    """Given a BDL sequence, computes the softmax for each time step."""
    b, d, l = seq.size()
    batch = seq.permute(0, 2, 1).contiguous().view(b*l, d)
    smbatch = F.softmax(batch)
    result = smbatch.view(b, l, d).permute(0, 2, 1).contiguous()
    return result

class Image2SeqTrainer(BasicTrainer):
    """Train image to sequence models using CTC.

    This takes images in BHWD order, plus output sequences
    consisting of lists of integers.
    """
    def __init__(self, *args, **kw):
        BasicTrainer.__init__(self, *args, **kw)

    def init_loss(self, loss=None):
        assert loss is None, "Image2SeqTrainer must be trained with BCELoss (default)"
        self.criterion = nn.BCELoss(size_average=False)

    def compute_loss(self, targets, weights=None):
        self.cutargets = None   # not used
        assert weights is None
        logits = self.cuoutput
        b, d, l = logits.size()
        probs = sequence_softmax(logits)
        assert dlh.sequence_is_normalized(probs), probs
        ttargets = torch.FloatTensor(targets)
        target_b, target_d, target_l = ttargets.size()
        assert b == target_b, (b, target_b)
        assert dlh.sequence_is_normalized(ttargets), ttargets
        aligned = ctc_align(probs.cpu(), ttargets.cpu())
        assert dlh.sequence_is_normalized(aligned)
        return self.criterion(probs, Variable(self._cuda(aligned)))

    def set_inputs(self, images):
        dlh.assign(self.cuinput, images, (0, 3, 1, 2))

    def set_targets(self, targets, outputs, weights=None):
        raise Exception("overridden by compute_loss")
