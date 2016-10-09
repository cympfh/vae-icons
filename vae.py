import argparse
import chainer
import chainer.functions as F
import glob
import numpy
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
        z_mu, z_var = self.enc(x, test)
        loss_kl = F.gaussian_kl_divergence(z_mu, z_var) / batch_size / self.k

        loss_decode = 0
        for _ in range(k):
            z = F.gaussian(z_mu, z_var)
            x_ = self.dec(z, test)
            loss_decode += F.mean_squared_error(x, x_) / k

        print(loss_kl.data, loss_decode.data)

        if not test:
            print('z_m', lib.debug.show(z_mu.data[0]))
            print('z_v', lib.debug.show(z_var.data[0]))
            # print('z ', lib.debug.show(z.data[0]))
            print('x  ', lib.debug.show(x.data[0][0][20]))
            print('x_ ', lib.debug.show(x_.data[0][0][20]))

        chainer.report({
            'loss_kl': loss_kl,
            'loss_decode': loss_decode
            }, self)

        return loss_kl + loss_decode * 0.1
        return loss_kl * 0.0002 + loss_decode


if __name__ == '__main__':

    k = 300

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

    images = []
    # images += glob.glob('./datasets/*.jpg')
    images += glob.glob('./datasets/*.png')
    images += glob.glob('./datasets/*.gif')
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

    batch_size = 128
    all_iter = chainer.iterators.SerialIterator(all_ds, batch_size)

    model = VAE(k)

    if args.gpu > -1:
        model.to_gpu()

    opt = chainer.optimizers.Adam()
    opt.setup(model)

    updater = chainer.training.StandardUpdater(all_iter, opt, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (10000, 'epoch'))
    trainer.extend(lib.training.Evaluate(evalfunc=test),
                   trigger=(100, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport([
        'main/loss_kl', 'main/loss_decode',
        'validation/main/loss_kl', 'validation/main/loss_decode']))

    trainer.run()
