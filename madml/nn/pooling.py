from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Union, List, Optional

from madml import tensor, zeros
from .module import Module
from .transform import vol2col
import numpy as np


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


class _MaxPoolNd(Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']

    return_indices: bool
    ceil_mode: bool

    def __init__(self, dims, kernel_size: Union[int, List[int]], stride: Union[int, List[int]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1, return_indices: bool = False,
                 ceil_mode: bool = False) -> None:
        super(_MaxPoolNd, self).__init__()
        self.dims = 3

        self.kernel_size = _dim_fix([1 for _ in range(self.dims)], kernel_size, dims)
        self.stride = _dim_fix([1 for _ in range(self.dims)], stride, dims)
        self.padding = _dim_fix([0 for _ in range(self.dims)], padding, dims)
        self.dilation = _dim_fix([1 for _ in range(self.dims)], dilation, dims)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self._col = []
        self._vol = []
        self.batch_size = 0
        self.in_channels = 1
        self.col = None
        self.kernel = None

    def forward_cpu(self, x: tensor) -> tensor:
        if self._col == [] or self._vol == []:
            self._col = [1 for _ in range(self.dims)]
            self._vol = [1 for _ in range(self.dims)]

            for i in range(self.dims - 1, 0, -1):
                self._col[i] = int(
                    (x.shape[i + 2] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) //
                    self.stride[i]) + 1
                self._vol[i] = x.shape[i + 2]
            self.batch_size = x.shape[0] * x.shape[1]
            self.kernel = vol2col(self.batch_size, self.in_channels, self._vol, self._col, self.kernel_size,
                                  self.stride, self.padding, self.dilation)
        y = zeros([x.shape[0], x.shape[1], *self._col])

        self.col = self.kernel.forward_cpu(x)
        max_idx = np.argmax(self.col.host_data, axis=0)
        y.host_data = self.col.host_data[max_idx, range(max_idx.size)]

        y.reshape([self.in_channels, self.batch_size, self._col[0], self._col[1], self._col[2], ])
        y.transpose([1, 0, 2, 3, 4])
        self.cache.append(x)
        self.cache.append(y)
        self.cache.append(max_idx)
        return y

    def backward_cpu(self) -> None:
        x, y, max_idx = self.cache
        dx, dy = x.gradient, y.gradient
        dy_col = dy.host_data.ravel()
        self.col.gradient.host_data[max_idx, range(dy_col.size)] = dy_col
        _ = self.kernel.backward_cpu()
        return x


class MaxPool1d(_MaxPoolNd):
    kernel_size: int
    stride: int
    padding: int
    dilation: int

    def __init__(self, kernel_size: int, stride: Optional[int] = None,
                 padding: int = 0, dilation: int = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool1d, self).__init__(1, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class MaxPool2d(_MaxPoolNd):
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]]
    padding: Union[int, List[int]]
    dilation: Union[int, List[int]]

    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool2d, self).__init__(2, kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class MaxPool3d(_MaxPoolNd):
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]]
    padding: Union[int, List[int]]
    dilation: Union[int, List[int]]

    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool3d, self).__init__(3, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
