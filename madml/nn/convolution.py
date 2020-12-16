from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .module import Module, Parameter
from typing import Union, List, Optional
from madml import tensor, xavier_uniform, zeros, zeros_like
from joblib import Parallel, delayed


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


def matmul(x: tensor, w: tensor, y: tensor, use_grad: bool = False):
    for m in range(x.shape[0]):
        for n in range(w.shape[1]):
            acc = 0
            for k in range(w.shape[0]):
                acc += x.host_data[m * w.shape[0] + k] * w.host_data[k * w.shape[1] + n]
            if use_grad:
                y.grad_data[m * w.shape[0] + n] = acc
            else:
                y.host_data[m * w.shape[0] + n] = acc


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
        self.vol = None
        self.bias = None
        if transposed:
            weight_shape = [in_channels, out_channels // groups, *self.kernel_size]
        else:
            weight_shape = [out_channels, in_channels // groups, *self.kernel_size]
        self.weight = Parameter(xavier_uniform(), weight_shape)

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache.append(x)
        if self._col == [] or self._vol == []:
            self._col = [1 for _ in range(self.dims)]
            self._vol = [1 for _ in range(self.dims)]

            for i in range(self.dims - 1, -1, -1):
                self._col[i] = int(
                    (x.shape[i + 2] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) //
                    self.stride[i]) + 1
                self._vol[i] = x.shape[i + 2]
            self.batch_size = x.shape[0]
        y = zeros([self.batch_size, self.out_channels, *self._col])
        self._2col(x.host_data)
        self.weight.reshape([-1, self.col.shape[0]])
        matmul(self.col, self.weight.param, y)

        if self._use_bias:
            if self.bias is None:
                self.bias = Parameter(zeros, y.shape[1:])
            bs = self.bias.size
            for b in range(y.shape[0]):
                for i in range(bs):
                    y.host_data[b * bs + i] += self.bias.param.host_data[i]

        return y

    def backward_cpu(self, dy: tensor) -> tensor:
        x = self.cache[0]
        dy_reshaped = dy.reshape([self.out_channels, -1])
        matmul(dy_reshaped, self.col, self.weight.param, True)
        matmul(self.weight.reshape([-1, self.out_channels]), dy_reshaped, x.gradient)

        self.vol.link(x)
        return x

    def _2col(self, x: List[Union[float, int, bytes, bool]]):
        n_output_plane = self.in_channels
        output_length = self.batch_size
        index_length = self.in_channels
        _col_size = 1

        for k in self.kernel_size:
            n_output_plane *= k
        for c in self._col:
            output_length *= c
            index_length *= c
            _col_size *= c

        if self.col is None:
            self.col = zeros([n_output_plane, output_length])

        for elt in range(self.batch_size):
            data_col = elt * self.in_channels * self._vol[0] * self._vol[1] * self._vol[2]
            data_vol = elt * n_output_plane * self._col[0] * self._col[1] * self._col[2]
            for index in range(index_length):
                w_offset = index % self.kernel_size[2]
                h_offset = (index / self.kernel_size[2]) % self.kernel_size[1]
                d_offset = (index / self.kernel_size[2] / self.kernel_size[1]) % self.kernel_size[0]
                c_vol = int(index / self.kernel_size[2] / self.kernel_size[1] / self.kernel_size[0])
                for d_col in range(self._col[0]):
                    d_vol = d_col * self.stride[0] - self.padding[0] + d_offset * self.dilation[0]
                    for h_col in range(self._col[1]):
                        h_vol = h_col * self.stride[1] - self.padding[1] + h_offset * self.dilation[1]
                        for w_col in range(self._col[2]):
                            w_vol = w_col * self.stride[2] - self.padding[2] + w_offset * self.dilation[2]
                            if (0 <= d_vol < self._vol[0] and 0 <= h_vol < self._vol[
                                1] and 0 <= w_vol < self._vol[2]):
                                data_vol_idx = data_vol + ((c_vol * self._vol[0] + d_vol) * self._vol[1] + h_vol) * \
                                               self._vol[2] + w_vol
                                data_col_idx = data_col + ((index * self._col[0] + d_col) * self._col[1] + h_col) * \
                                               self._col[2] + w_col
                                if data_vol_idx < len(x) and data_col_idx < self.col.size:
                                    self.col.host_data[int(data_col_idx)] = x[int(data_vol_idx)]

    def _2vol(self, x: List[Union[float, int, bytes, bool]]):
        n_output_plane = self.in_channels
        output_length = self.batch_size
        index_length = self.in_channels

        for k in self.kernel_size:
            n_output_plane *= k
        for c in self._col:
            output_length *= c
            index_length *= c

        for elt in range(self.batch_size):
            data_col = elt * self.in_channels * self._vol[0] * self._vol[1] * self._vol[2]
            data_vol = elt * n_output_plane * self._col[0] * self._col[1] * self._col[2]
            for index in range(index_length):
                w_offset = index % self.kernel_size[2]
                h_offset = (index / self.kernel_size[2]) % self.kernel_size[1]
                d_offset = (index / self.kernel_size[2] / self.kernel_size[1]) % self.kernel_size[0]
                c_vol = int(index / self.kernel_size[2] / self.kernel_size[1] / self.kernel_size[0])
                for d_col in range(self._col[0]):
                    d_vol = d_col * self.stride[0] - self.padding[0] + d_offset * self.dilation[0]
                    for h_col in range(self._col[1]):
                        h_vol = h_col * self.stride[1] - self.padding[1] + h_offset * self.dilation[1]
                        for w_col in range(self._col[2]):
                            w_vol = w_col * self.stride[2] - self.padding[2] + w_offset * self.dilation[2]
                            if (0 <= d_vol < self._vol[0] and 0 <= h_vol < self._vol[
                                1] and 0 <= w_vol < self._vol[2]):
                                data_vol_idx = data_vol + ((c_vol * self._vol[0] + d_vol) * self._vol[1] + h_vol) * \
                                               self._vol[2] + w_vol
                                data_col_idx = data_col + ((index * self._col[0] + d_col) * self._col[1] + h_col) * \
                                               self._col[2] + w_col
                                if data_col_idx < len(x) and data_vol_idx < self.vol.size:
                                    self.col.gradient.host_data[int(data_vol_idx)] += x[int(data_col_idx)]


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
