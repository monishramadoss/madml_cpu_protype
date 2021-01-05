from collections import defaultdict
from typing import Dict

from madml.nn import Parameter


class Optimizer(object):
    _use_velocity: bool

    def __init__(self, params: Dict[int, Parameter], defaults: dict) -> None:
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.params = params
        self._use_velocity = False

    def zero_grad(self) -> None:
        for _, p in self.params.items():
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: Dict[int, Parameter], lr: float = 1e-2, momentum: int = 0.9, dampening: int = 0,
                 weight_decay: float = 0, nesterov: bool = False) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None) -> None:
        for x, p in self.params.items():
            print(p.param.host_data.max(), end=' => ')
            for i in range(p.velocity.size):
                p.velocity.host_data.ravel()[i] = self.defaults['momentum'] * p.velocity.host_data.ravel()[i] \
                          + self.defaults['lr'] * p.param.gradient.host_data.ravel()[i]
                p.param.host_data.ravel()[i] -= p.velocity.host_data.ravel()[i]
            print(p.param.host_data.max())
        return
