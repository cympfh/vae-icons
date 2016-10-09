import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):

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
        h = F.sigmoid(self.dc4(h))
        return h
