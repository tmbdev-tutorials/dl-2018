from numpy import *
import numpy as np
from random import randint, uniform
from pylab import randn
import h5py
import torch
from torch import nn
from torch import optim
import torchvision
from scipy import ndimage as ndi
import flex
import layers
import time
import copy
import pylab
import matplotlib.pyplot as plt
from IPython import display

training_figsize = (4,4)


def evaluate(model, images, targets, bs=200):
    assert images.dtype == torch.float
    assert targets.dtype == torch.float
    losses = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            outputs = model.forward(images[i:i+bs].type(torch.float))
            results.append(outputs)
            loss = criterion(outputs, targets[i:i+bs])
            losses.append(float(loss))
    return mean(losses)

def train(model, images, targets, ntrain=100000, bs=20, lr=0.001, momentum=0.9, decay=0.0):
    assert images.dtype == torch.float
    assert targets.dtype == torch.float
    with torch.no_grad():
        model.forward(images[:bs].type(torch.float))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    losses = []
    for i in range(ntrain//bs):
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            start = randint(0, len(images)-bs)
            inputs = images[start:start+bs].type(torch.float)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], targets[start:start+bs].type(torch.float)) + \
                            outputs[1]
            else:
                loss = criterion(outputs, targets[start:start+bs].type(torch.float))
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
    return losses


class AutoMLP(object):
    def __init__(self, make_model, images, targets,
                 initial_bs=50,
                 initial_lrs=10**linspace(-6, 2, 20),
                 initial_ntrain=10000,
                 momentum=0.9,
                 ntrain=10000,
                 maxtrain=1e6,
                 selection_noise=0.05,
                 mintrain=1e6,
                 stop_no_improvement=2.0,
                 decay=1e-6
                 ):
        self.make_model = make_model
        self.images = images
        self.targets = targets
        self.verbose = False
        self.initial_bs = initial_bs
        self.initial_lrs = initial_lrs
        self.initial_ntrain = int(initial_ntrain)
        self.ntrain = int(ntrain)
        self.momentum = momentum
        self.decay = decay
        self.maxrounds = int(maxtrain) // int(ntrain)
        self.selection_noise = selection_noise
        self.mintrain = int(mintrain)
        self.stop_no_improvement = stop_no_improvement
        self.best_model = None

    def initial_population(self, make_model):
        population = []
        for lr in self.initial_lrs:
            model = make_model().cuda()
            model.PARAMS = dict(bs=self.initial_bs, lr=lr, id=randint(0, 1000000000))
            model.LOG = []
            population.append(model)
        return population

    def selection(self, population, size=4, key="training_loss"):
        if len(population) == 0: return []
        for model in population:
            model.KEY = (1.0+randn()*self.selection_noise) * model.LOG[-1][key]
        population = sorted(population, key=lambda m: m.KEY)
        while len(population) > size: del population[-1]
        return population

    def mutation(self, old_population, variants=1, variation=0.8):
        population = []
        for model in old_population:
            population += [model]
            for _ in range(variants):
                cloned = copy.deepcopy(model)
                cloned.PARAMS = dict(
                    lr = clip(cloned.PARAMS["lr"] * uniform(variation, 1.0/variation), 1e-7, 1e2),
                    bs = clip(int(cloned.PARAMS["bs"] * uniform(variation, 1.0/variation)), 1, 1000),
                    id = randint(0, 1000000000)
                )
                population += [cloned]
        return population

    def train_population(self, population, ntrain=50000, momentum=0.9, verbose=False):
        infos = []
        for model in population:
            lr, bs = [model.PARAMS[name] for name in "lr bs".split()]
            ntrained = 0 if len(model.LOG)==0 else model.LOG[-1]["ntrain"]
            training_loss = train(model, self.images, self.targets, lr=lr, bs=bs,
                                  momentum=self.momentum,
                                  decay=self.decay,
                                  ntrain=ntrain)
            if isinstance(training_loss, list):
                training_loss = mean(training_loss[-100:])
            info = dict(
                    training_loss=training_loss,
                    lr=lr,
                    ntrain=ntrain+ntrained,
                    momentum=momentum,
                    bs=bs)
            if self.verbose: print info
            model.LOG += [info]
            infos += [info]
        return infos

    def to(self, device):
        for model in self.population:
            model.to(device)

    def cpu(self):
        for model in self.population:
            model.cpu()

    def train(self):
        self.fig = plt.figure(figsize=training_figsize)
        self.fig.add_subplot(1,1,1)
        self.ax = self.fig.get_axes()[0]        
        self.infos = []
        self.population = self.initial_population(self.make_model)
        initial_infos = self.train_population(self.population,  ntrain=self.initial_ntrain)
        # infos += initial_infos
        self.population = self.selection(self.population)
        for r in xrange(self.maxrounds):
            old_population = [copy.deepcopy(model) for model in self.population]
            self.population = self.mutation(self.population)
            self.infos += self.train_population(self.population, ntrain=self.ntrain)
            self.population = self.selection(self.population + old_population)
            # information display
            if len(self.infos)>0: self.display()
        self.cpu()
        self.fig.clf()
        return self.best()

    def display(self, key="training_loss", yscale="log", ylim=None):
        self.ax.cla()
        self.ax.set_yscale(yscale)
        if ylim is not None: self.ax.set_ylim(ylim)
        self.ax.scatter(*zip(*[(l["ntrain"], l["training_loss"]) for l in self.infos]))
        display.clear_output(wait=True)
        display.display(self.fig)
        
    def best(self):
        return self.population[0].cuda()

    def info(self, model=None):
        for i, key in enumerate("training_loss lr bs".split()):
            pylab.subplot(2, 2, i+1)
            pylab.plot(*zip(*[(l["ntrain"], l[key]) for l in model.LOG]))
