import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training.extensions
import glob
import numpy
import sys
from PIL import Image

import lib.datasets
import lib.training
import lib.debug


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--save', action='store_true', help="model save per 10epoch")
parser.add_argument('--skip-validation', action='store_true', help="skip validation")
args = parser.parse_args()


xp = numpy

if args.gpu > -1:
    xp = chainer.cuda.cupy
    chainer.cuda.get_device(args.gpu).use()


class Encoder(chainer.Chain):

    def __init__(self, k):
        """ k = dim(z) """
        super().__init__(
            c1=L.Convolution2D(3, 16, 4, stride=2),
            c2=L.Convolution2D(16, 32, 5, stride=2),
            c3=L.Convolution2D(32, 64, 4, stride=2),
            c4=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            bn1=L.BatchNormalization(16),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(64),
            out_mu=L.Linear(512, k),
            out_var=L.Linear(512, k),
        )

    def __call__(self, x, test=False):
        h = self.bn1(F.elu(self.c1(x)), test=test)
        h = self.bn2(F.elu(self.c2(h)), test=test)
        h = self.bn3(F.elu(self.c3(h)), test=test)
        h = F.elu(self.c4(h))
        mu = self.out_mu(h)
        var = self.out_var(h)
        return mu, var


class Decoder(chainer.Chain):

    def __init__(self, k):
        """ k = dim(z) """
        self.k = k
        super().__init__(
            lin=L.Linear(k, 128 * 2 * 2),
            dc1=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(64, 32, 4, stride=2),
            dc3=L.Deconvolution2D(32, 16, 5, stride=2),
            dc4=L.Deconvolution2D(16, 3, 4, stride=2),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(32),
            bn3=L.BatchNormalization(16),
        )

    def __call__(self, z, test=False):
        h = F.reshape(self.lin(z), (z.data.shape[0], 128, 2, 2))
        h = self.bn1(F.elu(self.dc1(h)), test=test)
        h = self.bn2(F.elu(self.dc2(h)), test=test)
        h = self.bn3(F.elu(self.dc3(h)), test=test)
        x = F.sigmoid(self.dc4(h))
        return x


class VAE(chainer.Chain):

    def __init__(self, k=512):
        self.k = k
        super().__init__(
            enc=Encoder(k),
            dec=Decoder(k)
        )

    def __call__(self, x, test=False, k=32):

        mu, var = self.enc(x, test)
        loss_kl = F.gaussian_kl_divergence(mu, var) / k

        loss_decode = 0
        for _ in range(k):
            z = F.gaussian(mu, var)
            x_hat = self.dec(z, test)
            loss_decode += F.mean_squared_error(x, x_hat)
            # loss_decode += F.bernoulli_nll(x, x_hat)

        print(loss_kl.data, loss_decode.data)

        if not test:
            print('x', lib.debug.show(x.data[0][0][20]))
            print('m', lib.debug.show(mu.data[0]))
            print('v', lib.debug.show(var.data[0]))
            print('z', lib.debug.show(z.data[0]))
            print('x', lib.debug.show(x_hat.data[0][0][20]))

        return loss_kl * 0.001 + loss_decode


if __name__ == '__main__':

    k = 100

    test_ds = lib.datasets.ImageDataset(['./datasets/cympfh.png'])

    def test(t):
        """test generating"""
        x = xp.array(test_ds[0]).reshape(1, 3, 48, 48).astype('f')
        z, _ = model.enc(x, test=True)
        x = model.dec(z, test=True).data * 256
        data = x[0].transpose(1, 2, 0)
        data = data.astype(numpy.uint8)
        data = chainer.cuda.to_cpu(data)
        data = Image.fromarray(data)
        data.save("ae.{:03d}.png".format(t))

        z = chainer.Variable(xp.random.normal(size=(1, k)).astype('f'))
        x = model.dec(z, test=True).data * 256
        data = x[0].transpose(1, 2, 0)
        data = data.astype(numpy.uint8)
        data = chainer.cuda.to_cpu(data)
        data = Image.fromarray(data)
        data.save("rand.{:03d}.png".format(t))


    def save(t):
        """save the model"""
        filename = "model.{:03d}.npz".format(t)
        chainer.serializers.save_npz(filename, model)


    images = glob.glob('./datasets/*.jpg')
    images += glob.glob('./datasets/*.png')
    images += glob.glob('./datasets/*.gif')
    # images = images[-256:]
    all_ds = lib.datasets.ImageDataset(images)

    # validation
    if not args.skip_validation:
        sys.stderr.write("data validation...\n")
        for i in range(len(images)):
            sys.stderr.write("\r {} / {}   ".format(i + 1, len(images)))
            try:
                img = all_ds.get_example(i)
            except:
                sys.stderr.write("\n[ERR] <path={} index={}> has wrong".format(images[i], i))
                sys.exit()
        sys.stderr.write("ok\n")

    batch_size = 128
    all_iter = chainer.iterators.SerialIterator(all_ds, batch_size, shuffle=False)

    model = VAE(k)

    if args.gpu > -1:
        model.to_gpu()

    opt = chainer.optimizers.Adam()
    opt.setup(model)

    updater = chainer.training.StandardUpdater(all_iter, opt, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (1000, 'epoch'), out='result')
    trainer.extend(chainer.training.extensions.LinearShift('alpha', (0.002, 0.0001), (2, 3000)))
    trainer.extend(lib.training.Evaluate(evalfunc=test), trigger=(100, 'iteration'))
    if args.save:
        trainer.extend(lib.training.Evaluate(evalfunc=save), trigger=(100, 'epoch'))

    trainer.run()
