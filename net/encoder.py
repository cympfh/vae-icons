import chainer
import chainer.functions as F
import chainer.links as L


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
