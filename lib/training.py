from chainer.training import extension


class Evaluate(extension.Extension):

    def __init__(self, evalfunc):
        self.t = 0
        self.evalfunc = evalfunc

    def __call__(self, trainer):
        self.evalfunc(self.t)
        self.t += 1
