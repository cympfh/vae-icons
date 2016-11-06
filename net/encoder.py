import chainer
import chainer.functions as F
import chainer.links as L


class Encoder(chainer.Chain):

    def __init__(self, k):
        """ k = dim(z) """
        super().__init__(
            c1=L.Convolution2D(3, 32, 3, pad=1),
            c2=L.Convolution2D(32, 64, 3, pad=1),
            c3=L.Convolution2D(64, 256, 3, pad=1),
            c4=L.Convolution2D(256, 512, 3, pad=1),
            c5_mu=L.Convolution2D(512, 512, 4, stride=4),
            c5_vr=L.Convolution2D(512, 512, 4, stride=4),
            out_mu=L.Linear(512, k),
            out_vr=L.Linear(512, k),
            bn1=L.BatchNormalization(32),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(256),
            bn4=L.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        h = x
        h = F.elu(self.bn1(self.c1(h), test=test))
        h = F.max_pooling_2d(h, 2)
        h = F.elu(self.bn2(self.c2(h), test=test))
        h = F.max_pooling_2d(h, 2)
        h = F.elu(self.bn3(self.c3(h), test=test))
        h = F.max_pooling_2d(h, 2)
        h = F.elu(self.bn4(self.c4(h), test=test))
        h = F.average_pooling_2d(h, 2)
        mu = self.out_mu(F.sigmoid(self.c5_mu(h)))
        vr = self.out_vr(F.sigmoid(self.c5_vr(h)))
        return mu, vr
