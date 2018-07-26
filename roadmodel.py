from numpy import *
from pylab import rand, randn, randint
import scipy
import scipy.ndimage as ndi


def spatial_sampler(image):
    h, w = image.shape
    probs = image.reshape(h*w)
    assert amin(probs) >= 0
    probs = probs * 1.0 / sum(probs)
    probs = add.accumulate(probs)
    def f():
        x = rand()
        i = searchsorted(probs, x)
        return (i//w, i%w)
    return f

class RoadModel(object):
    def __init__(self, n=256, background=0.01, sigma=10.0):
        self.n = n
        self.background = background
        self.minsize = 3
        self.scale = 0.05

        obstacle_prior = zeros([n, n])
        xs, ys = meshgrid(linspace(-1, 1, n), linspace(-1, 1, n))
        prior = 1.0 * maximum((xs>0.5+0.5*ys), (xs<-0.5-0.5*ys))
        self.road_map = prior

        prior = ndi.gaussian_filter(prior, sigma)
        prior -= amin(prior); prior /= amax(prior)
        prior = maximum(prior, self.background)
        self.prior = prior

        self.xs, self.ys = meshgrid(range(n), range(n))
        self.sampler = spatial_sampler(self.prior)

    def sampled_image(self, k=100000):
        n = self.n
        result = zeros([n, n])
        for _ in xrange(k):
            i, j = self.sampler()
            result[i, j] += 1.0
        return result

    def sample(self, k):
        if isinstance(k, tuple): k = randint(*k)
        return [self.sampler() for _ in range(k)]

    def render(self, samples):
        n = self.n
        target = zeros([n, n], 'f')
        for i, j in samples:
            target[i, j] = 1.0
        d = ndi.distance_transform_cdt(target==0)
        target = 1.0*(d < maximum(self.minsize, self.ys*self.scale))
        return target

    def road_truth(self, k=(5, 20)):
        return self.render(self.sample(k))

    def sense(self, target, fg=0.2, bg=0.02):
        n = self.n
        return 1.0*maximum(rand(n, n) < fg * target, rand(n, n) < bg)
