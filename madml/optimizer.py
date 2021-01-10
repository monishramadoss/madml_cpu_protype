from collections import defaultdict
from typing import Dict, List
import numpy as np

from madml import tensor
from madml.nn import Parameter
from .init import zeros_like


def print_p(p: Parameter) -> None:
    print('\tdata', p.param.host_data.max(),
          '\n\tgrad', p.param.gradient.host_data.max(),
          '\n\tvelo', p.velocity.host_data.max())


class Optimizer(object):
    _use_velocity: bool

    def __init__(self, params: Dict[int, Parameter], defaults: dict) -> None:
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.params = params
        self._use_velocity = False

    def zero_grad(self) -> None:
        print('parameter#', len(self.params.items()))
        for _, t in self.params.items():
            t.zero_grad()

    def step(self):
        raise NotImplementedError


def dl1_reg(w: np.ndarray, lam: float = 1e-3, esp: float = 1e-8) -> np.ndarray:
    w *= lam
    w /= np.abs(w + esp)
    return w


def dl2_reg(w: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    w *= lam
    return w

class SGD(Optimizer):
    def __init__(self, params: Dict[int, Parameter], lr: float = 1e-3, momentum: float = 0.0, dampening: int = 0,
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
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])
            print_p(p)
            p.velocity.host_data = self.defaults['momentum'] * p.velocity.host_data \
                                   + self.defaults['lr'] * p.param.gradient.host_data
            p.param.host_data -= p.velocity.host_data


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


class Adam(Optimizer):
    def __init__(self, params: Dict[int, Parameter], lr: float = 1e-3, betas: List[float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.M = {k: zeros_like(v.param) for k, v in params.items()}
        self.counter = 1
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None) -> None:
        for k, p in self.params.items():
            p.param.reset_shape()
            p.param.gradient.host_data += dl2_reg(p.param.host_data, self.defaults['lr'])
            m = self.M[k].host_data
            r = p.velocity.host_data
            self.M[k].host_data = exp_running_avg(m, p.param.host_data, self.defaults['betas'][0])
            p.velocity.host_data = exp_running_avg(r, p.param.host_data ** 2, self.defaults['betas'][1])
            m_k_hat = self.M[k].host_data / (1. - self.defaults['betas'][0] ** self.counter)
            r_k_hat = p.velocity.host_data / (1. - self.defaults['betas'][1] ** self.counter)
            p.param.host_data -= self.defaults['lr'] * m_k_hat / np.sqrt(r_k_hat) + self.defaults['eps']
            print_p(p)
        self.counter += 1
