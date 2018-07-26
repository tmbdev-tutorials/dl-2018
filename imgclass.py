from numpy import *
import numpy as np
from random import randint, uniform
from pylab import randn
import h5py
import torch
from torch import nn
from torch import optim
import numpy as np
import torchvision
from scipy import ndimage as ndi
import flex
import layers
import time
import copy
import pylab
from IPython import display
import matplotlib.pyplot as plt

training_figsize = (4, 4)

def one_hot(classes, nclasses=None, value=1.0):
    if nclasses is None: nclasses = 1+np.amax(classes)
    targets = torch.FloatTensor(len(classes), nclasses)
    targets[:, :] = 0
    return targets.scatter(1, classes.reshape(-1, 1), value)

def C(images):
    if isinstance(images, np.ndarray):
        raise Error("accepts only Torch tensors")
    if images.dtype == torch.uint8:
        return images.type(torch.float)/255.0
    elif images.dtype == torch.float:
        return images
    else:
        raise Error("unknown dtype", images.dtype)
        
def evaluate(model, images, classes, bs=200, return_results=False):
    results = []
    errs = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), bs):
            outputs = model.forward(C(images[i:i+bs]))
            _, indexes = outputs.max(1)
            results.append(indexes)
            errs += int((indexes!=classes[i:i+bs]).sum())
            total += outputs.size(0)
    erate = float(errs) / total
    if return_results: return erate, total, results
    return erate

def train(model, images, classes, ntrain=100000, bs=20, lr=0.001, momentum=0.9, decay=0, mode="ce"):

    with torch.no_grad():
        model.forward(C(images[:bs]))
    if mode.lower() in ["ce", "crossentropy"]:
        criterion = nn.CrossEntropyLoss()
        expand = lambda target, output: target
    elif model.lower() in ["mse", "meansquared", "meansquarederror"]:
        criterion = nn.MSELoss()
        expand = lambda target, output: one_hot(target, output.size(1))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    losses = []
    for i in range(ntrain//bs):
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            start = randint(0, len(images)-bs)
            inputs = C(images[start:start+bs])
            outputs = model(inputs)
            target = expand(classes[start:start+bs], outputs)
            loss = criterion(outputs, target)
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
    return mean(losses[-100:])

class AutoMLP(object):
    def __init__(self, make_model, images, classes, test_images, test_classes,
                 initial_bs=50,
                 initial_lrs=10**linspace(-6, 2, 20),
                 initial_ntrain=100000,
                 momentum=0.9,
                 decay=0,
                 ntrain=50000,
                 maxtrain=3e6,
                 selection_noise=0.05,
                 mintrain=1e6,
                 stop_no_improvement=2.0
                 ):
        self.make_model = make_model
        self.images = images
        self.classes = classes
        self.test_images = test_images
        self.test_classes = test_classes
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

    def is_better(self, model, other):
        if model is None: return False
        if other is None: return True
        return model.LOG[-1]["test_loss"] < other.LOG[-1]["test_loss"]

    def train_population(self, population, ntrain=50000, momentum=0.9, verbose=False):
        infos = []
        for model in population:
            lr, bs = [model.PARAMS[name] for name in "lr bs".split()]
            ntrained = 0 if len(model.LOG)==0 else model.LOG[-1]["ntrain"]
            training_loss = train(model, self.images, self.classes, lr=lr, bs=bs,
                                  momentum=self.momentum, ntrain=ntrain, decay=self.decay)
            test_loss = evaluate(model, self.test_images, self.test_classes)
            info = dict(
                    training_loss=training_loss,
                    test_loss=test_loss,
                    lr=lr,
                    ntrain=ntrain+ntrained,
                    momentum=momentum,
                    bs=bs)
            if self.verbose: print info
            model.LOG += [info]
            infos += [info]
            if self.is_better(model, self.best_model):
                self.best_model = model
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
        self.best_model = None
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
            maxtrained = amax([m.LOG[-1]["ntrain"] for m in self.population])
            l = self.best_model.LOG[-1]
            print "#best", l["test_loss"], "@", l["ntrain"], "of", maxtrained
            last_best = l["ntrain"]
            if maxtrained > self.mintrain and maxtrained > self.stop_no_improvement * maxtrained:
                print "# stopping b/c no improvement"
                break
        self.cpu()
        return self.population[0].cpu()

    def display(self, key="training_loss", yscale="log", ylim=None):
        self.ax.cla()
        self.ax.set_yscale(yscale)
        if ylim is not None: self.ax.set_ylim(ylim)
        self.ax.scatter(*zip(*[(l["ntrain"], l["training_loss"]) for l in self.infos]))
        display.clear_output(wait=True)
        display.display(self.fig)
