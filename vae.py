import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import glob
import numpy
from PIL import Image
from imagedataset import ImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()


xp = numpy

if args.gpu > -1:
    xp = chainer.cuda.cupy
    chainer.cuda.get_device(args.gpu).use()


class Encoder(chainer.Chain):

    def __init__(self, k=100):
        """ k = dim(z) """
        super().__init__(
            a1=L.Convolution2D(3, 16, (5, 1), stride=(2, 1), pad=(2, 0)),
            a2=L.Convolution2D(16, 16, (1, 5), stride=(1, 2), pad=(0, 2)),
            b=L.Convolution2D(3, 16, 1),
            c=L.Convolution2D(16, 64, 5, stride=2),
            d=L.Convolution2D(64, 128, 2),
            e=L.Convolution2D(128, 256, 2),
            bn0=L.BatchNormalization(3),
            bn1=L.BatchNormalization(16),
            out_mu=L.Linear(256, k),
            out_sigma=L.Linear(256, k),
        )

    def __call__(self, x):
        x = (x - 128) / 256
        x = self.bn0(x)
        h1 = F.elu(self.bn1(self.a2(self.a1(x))))
        h2 = F.average_pooling_2d(self.b(x), 2)
        h = h1 * 0.8 + h2 * 0.2
        h = F.elu(self.c(h))
        h = F.elu(self.d(h))
        h = F.tanh(self.e(h))
        h = F.average_pooling_2d(h, 8)
        mu = self.out_mu(h)
        sigma = self.out_sigma(h)
        return mu, sigma


class Decoder(chainer.Chain):

    def __init__(self, k=100):
        """ k = dim(z) """
        self.k = k
        super().__init__(
            lin=L.Linear(k, 128 * 2 * 2, wscale=0.001),
            a=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            b=L.Deconvolution2D(64, 32, 4, stride=2),
            c=L.Deconvolution2D(32, 16, 5, stride=2),
            d=L.Deconvolution2D(16, 3, 4, stride=2),
        )

    def __call__(self, z):
        h = F.reshape(self.lin(z), (z.data.shape[0], 128, 2, 2))
        h = F.elu(self.a(h))
        h = F.elu(self.b(h))
        h = F.elu(self.c(h))
        h = F.elu(self.d(h))
        return h


class VAE(chainer.Chain):

    def __init__(self, k=100):
        self.k = k
        super().__init__(
            enc=Encoder(k),
            dec=Decoder(k)
        )

    def __call__(self, x):
        mu, sigma = self.enc(x)
        loss_kl = F.gaussian_kl_divergence(mu, sigma)
        z = mu + xp.random.normal() * F.exp(-sigma / 2)  # random sampling
        x_hat = self.dec(z)
        loss_decode = F.mean_squared_error(x, x_hat)
        loss = loss_kl + loss_decode
        print(loss_kl.data, loss_decode.data)
        return loss


def save_as_image(x, filename):
    if isinstance(x, chainer.Variable):
        x = x.data
    if isinstance(x, chainer.cuda.cupy.ndarray):
        x = chainer.cuda.to_cpu(x)
    data = (x * 256).reshape((3, 48, 48)).transpose(2, 1, 0)
    data = data.astype(numpy.uint8)
    data = Image.fromarray(data)
    data.save(filename)



images = glob.glob('./datasets/*.jpg')
all_ds = ImageDataset(images)
all_iter = chainer.iterators.SerialIterator(all_ds, 32)

model = VAE()
if args.gpu > -1:
    model.to_gpu()
opt = chainer.optimizers.Adam()
opt.setup(model)

updater = chainer.training.StandardUpdater(all_iter, opt, device=args.gpu)
trainer = chainer.training.Trainer(updater, (20, 'epoch'), out='result')
trainer.run()
