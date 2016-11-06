import argparse
import chainer
import chainer.functions as F
import glob
import numpy
import random
import sys
from PIL import Image
from chainer.training import extensions

import lib.datasets
import lib.training
import lib.debug
from net.encoder import Encoder
from net.gen import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--skip-validation', action='store_true')
args = parser.parse_args()


xp = numpy

if args.gpu > -1:
    xp = chainer.cuda.cupy
    chainer.cuda.get_device(args.gpu).use()


class VAE(chainer.Chain):

    def __init__(self, k=512):
        self.k = k
        super().__init__(
            enc=Encoder(k),
            dec=Generator(k)
        )

    def __call__(self, x, test=False, k=4):

        batch_size = x.data.shape[0]
        w = x.data.shape[2]
        tr, tg, tb = chainer.functions.split_axis(x, 3, 1)
        tr = F.reshape(tr, (batch_size * w * w, ))
        tg = F.reshape(tg, (batch_size * w * w, ))
        tb = F.reshape(tb, (batch_size * w * w, ))

        x = chainer.Variable(x.data.astype('f'))

        z_mu, z_var = self.enc(x, test)
        loss_kl = F.gaussian_kl_divergence(z_mu, z_var) / batch_size / self.k

        loss_decode = 0
        for _ in range(k):
            z = F.gaussian(z_mu, z_var)
            r, g, b = self.dec(z, test)
            r = F.transpose(r, (0, 2, 3, 1))
            r = F.reshape(r, (batch_size * w * w, 256))
            g = F.transpose(g, (0, 2, 3, 1))
            g = F.reshape(g, (batch_size * w * w, 256))
            b = F.transpose(b, (0, 2, 3, 1))
            b = F.reshape(b, (batch_size * w * w, 256))
            loss_decode += F.softmax_cross_entropy(r, tr) / k
            loss_decode += F.softmax_cross_entropy(g, tg) / k
            loss_decode += F.softmax_cross_entropy(b, tb) / k

        chainer.report({
            'loss_kl': loss_kl,
            'loss_decode': loss_decode
            }, self)

        beta = 0.2
        return beta * loss_kl + (1 - beta) * loss_decode


if __name__ == '__main__':

    k = 200
    w = 64

    test_ds = lib.datasets.ImageDataset(['./lib/cympfh.png'])

    def test(t):
        """test generating"""
        x = xp.array(test_ds[0]).reshape(1, 3, w, w).astype('f')
        z, _ = model.enc(x, test=True)
        rgb = list(model.dec(z, test=True))
        for j in range(3):
            rgb[j] = F.transpose(rgb[j], (0, 2, 3, 1))
            rgb[j] = F.reshape(rgb[j], (w * w, 256))
            rgb[j] = numpy.argmax(rgb[j].data, axis=1)
            rgb[j] = rgb[j].reshape((w, w))
            rgb[j] = chainer.cuda.to_cpu(rgb[j])
        img = numpy.stack(rgb, axis=2).astype(numpy.uint8)
        img = Image.fromarray(img)
        img.save("ae.png")

        z = chainer.Variable(xp.random.normal(size=(1, k)).astype('f'))
        rgb = list(model.dec(z, test=True))
        for j in range(3):
            rgb[j] = F.transpose(rgb[j], (0, 2, 3, 1))
            rgb[j] = F.reshape(rgb[j], (w * w, 256))
            rgb[j] = numpy.argmax(rgb[j].data, axis=1)
            rgb[j] = rgb[j].reshape((w, w))
            rgb[j] = chainer.cuda.to_cpu(rgb[j])
        img = numpy.stack(rgb, axis=2).astype(numpy.uint8)
        img = Image.fromarray(img)
        img.save("rand.png")

    print('load dataset')
    images = glob.glob('./dataset/*.png')
    print(len(images))
    random.shuffle(images)
    all_ds = lib.datasets.ImageDataset(images)

    # validation
    if not args.skip_validation:
        sys.stderr.write("data validation...\n")
        for i in range(len(images)):
            sys.stderr.write("\r {} / {}   ".format(i + 1, len(images)))
            try:
                img = all_ds.get_example(i)
            except:
                sys.stderr.write("\n[ERR] <path={} index={}> has wrong".format(
                    images[i], i))
                sys.exit()
        sys.stderr.write("ok\n")

    batch_size = 32
    all_iter = chainer.iterators.SerialIterator(all_ds, batch_size)

    model = VAE(k)

    if args.gpu > -1:
        model.to_gpu()

    opt = chainer.optimizers.Adam()
    opt.setup(model)

    m = 20
    updater = chainer.training.StandardUpdater(all_iter, opt, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (1000, 'epoch'))
    trainer.extend(lib.training.Evaluate(evalfunc=test), trigger=(m, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(m, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss_kl', 'main/loss_decode']))
    trainer.extend(extensions.snapshot_object(model, '{.updater.epoch}.npz'), trigger=(1, 'epoch'))
    trainer.run()
