import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):

    def __init__(self, k):
        """ k = dim(z) """
        self.k = k
        super().__init__(
            lin=L.Linear(k, 32 * 4 * 4),
            dc1=L.Deconvolution2D(32, 512, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(512, 512, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(512, 512, 4, stride=2, pad=1),
            norm0=L.BatchNormalization(32),
            norm1=L.BatchNormalization(512),
            norm2=L.BatchNormalization(512),
            norm3=L.BatchNormalization(512),
            dc_r=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            dc_g=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            dc_b=L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
        )

    def __call__(self, z, test=False):
        h = F.reshape(self.lin(z), (z.data.shape[0], 32, 4, 4))
        h = self.norm0(h)
        h = F.relu(self.norm1(self.dc1(h), test=test))
        h = F.relu(self.norm2(self.dc2(h), test=test))
        h = F.relu(self.norm3(self.dc3(h), test=test))
        r = self.dc_r(h)
        g = self.dc_g(h)
        b = self.dc_b(h)
        return r, g, b
