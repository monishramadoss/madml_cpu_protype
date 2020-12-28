from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union, List, Optional

from madml import tensor, xavier_uniform, zeros
from .module import Module, Parameter
from .transform import transpose, vol2col
import numpy as np

from numba import prange


def _dim_fix(arr, arg_arr, pi):
    def parse(x):
        return [x for _ in range(pi)] if isinstance(x, int) else [x[t] for t in range(pi)]

    if isinstance(arg_arr, int):
        arg_arr = parse(arg_arr)
    j = 0
    for i in range(len(arg_arr) - 1, len(arr)):
        arr[i] = arg_arr[j]
        j += 1
    return arr


def matmul(x: tensor, w: tensor, y: tensor):
    assert (x.shape[1] == w.shape[0])
    for m in range(x.shape[0]):
        for n in range(w.shape[1]):
            acc = 0
            for k in range(w.shape[0]):
                acc += x.host_data[m * w.shape[0] + k] * w.host_data[k * w.shape[1] + n]
            y.host_data[m * w.shape[1] + n] = acc


class ConvNd(Module):
    __constants__ = ['dims', 'stride', 'padding', 'dilation', 'groups', 'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[List]}
    dims: int
    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    transposed: bool
    output_padding: List[int]
    groups: int
    padding_mode: str
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, dims, in_channels: int, out_channels: int, kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]], padding: Union[int, List[int]], dilation: Union[int, List[int]],
                 transposed: bool, output_padding: Union[int, List[int]],
                 groups: int, bias: bool, padding_mode: str) -> None:
        super(ConvNd, self).__init__()

        if groups != 1:
            raise NotImplementedError("dilation not implemented in conv")

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}  # , 'reflect', 'replicate', 'circular'} # needs to be implemented
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes, padding_mode))
        self.dims = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _dim_fix([1 for _ in range(self.dims)], kernel_size, dims)
        self.stride = _dim_fix([1 for _ in range(self.dims)], stride, dims)
        self.padding = _dim_fix([0 for _ in range(self.dims)], padding, dims)
        self.dilation = _dim_fix([1 for _ in range(self.dims)], dilation, dims)
        self.transposed = transposed
        self.output_padding = _dim_fix([0 for _ in range(self.dims)], output_padding, dims)
        self.groups = groups
        self.padding_mode = padding_mode
        self._col = []
        self._vol = []
        self.params = []
        self._use_bias = bias
        self.batch_size = 1
        self.col = None
        self.bias = None
        if transposed:
            weight_shape = [in_channels, out_channels // groups, *self.kernel_size]
        else:
            weight_shape = [out_channels, in_channels // groups, *self.kernel_size]
        self.weight = Parameter(xavier_uniform(), weight_shape)
        self.kernel = None

    def forward_cpu(self, x: tensor) -> tensor:
        if self._col == [] or self._vol == []:
            self._col = [1 for _ in range(self.dims)]
            self._vol = [1 for _ in range(self.dims)]

            for i in range(self.dims - 1, -1, -1):
                self._col[i] = int(
                    (x.shape[i + 2] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) //
                    self.stride[i]) + 1
                self._vol[i] = x.shape[i + 2]
            self.batch_size = x.shape[0]

            self.kernel = vol2col(self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size,
                                  self.stride, self.padding, self.dilation)
            if self._use_bias and self.bias is not None:
                self.bias = Parameter(zeros, [self.out_channels, *self._col])
        y = zeros([self.batch_size, self.out_channels, *self._col])

        self.col = self.kernel.forward_cpu(x)
        self.weight.param.reshape([self.weight.param.shape[0], -1])
        y.host_data = self.weight.param.host_data @ self.col.host_data
        y.reshape([self.out_channels, self.batch_size, self._col[0], self._col[1], self._col[2], ])
        y.transpose([1, 0, 2, 3, 4])
        if self._use_bias and self.bias is not None:
            y.host_data += self.bias.param.host_data

        self.cache.append(x)
        self.cache.append(y)
        return y

    def backward_cpu(self) -> tensor:
        x, y = self.cache
        dx, dy = x.gradient, y.gradient
        dc = self.col.gradient
        assert (x.size == dx.size and dy.size == y.size and dc == self.col.gradient)
        if self.bias is not None:
            self.bias.param.gradient.host_data = np.sum(dy.host_data, axis=0)

        dy_reshaped = dy.host_data.transpose([1, 0, 2, 3, 4]).reshape(self.out_channels, -1)
        self.weight.param.gradient.host_data = dy_reshaped @ self.col.host_data.T
        self.weight.param.gradient.reset()

        w_reshaped = self.weight.param.host_data.reshape([self.out_channels, -1])
        dc.host_data = w_reshaped.T @ dy_reshaped
        _ = self.kernel.backward_cpu()

        return x


class Conv1d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]] = 1,
                 padding: Union[int, List[int]] = 0,
                 dilation: Union[int, List[int]] = 1,
                 groups: Union[int, List[int]] = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros'):
        super(Conv1d, self).__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                     groups, bias, padding_mode)


class Conv2d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]] = 1,
                 padding: Union[int, List[int]] = 0,
                 dilation: Union[int, List[int]] = 1,
                 groups: Union[int, List[int]] = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros'):
        super(Conv2d, self).__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                     groups, bias, padding_mode)


class Conv3d(ConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int]],
                 stride: Union[int, List[int]] = 1,
                 padding: Union[int, List[int]] = 0,
                 dilation: Union[int, List[int]] = 1,
                 groups: Union[int, List[int]] = 1,
                 bias: bool = False,
                 padding_mode: str = 'zeros'):
        super(Conv3d, self).__init__(3, in_channels, out_channels, kernel_size, stride, padding, dilation, False, 0,
                                     groups, bias, padding_mode)
