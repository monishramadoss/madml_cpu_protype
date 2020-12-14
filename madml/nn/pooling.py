from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .module import Module, Parameter
from typing import Union, List, Optional
from madml import tensor, xavier_uniform, zeros, zeros_like



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
        self.vol = None
        self.col = None

    def forward_cpu(self, x: tensor) -> tensor:
        self.cache.append(x)
        if self._col == [] or self._vol == []:
            self._col = [1 for _ in range(self.dims)]
            self._vol = [1 for _ in range(self.dims)]

            for i in range(self.dims - 1, 0, -1):
                self._col[i] = int((x.shape[i+2] + 2 * self.padding[i] - self.dilation[i] * (self.kernel_size[i] - 1) - 1) // self.stride[i]) + 1
                self._vol[i] = x.shape[i + 2]
            self.batch_size = x.shape[0]
        self._2col(x.host_data)

        max_idx = []
        self.cache = [x, max_idx]
        y = zeros([x.shape[0], x.shape[1], *self._col])

        return y

    def backward_cpu(self, dy: tensor) -> tensor:
        x, max_idx = self.cache
        dx_col = zeros(self.col.shape)

        # dy_col = np.transpose(dy, (2, 3, 4, 0, 1)).ravel()  # (72128,)
        # dx_col[max_idx, range(dy_col.size)] = dy_col

        self._2vol(dx_col.host_data)
        self.vol.link(x)
        return self.vol

    def _2col(self, x: List[Union[float, int, bytes, bool]]):
        n_output_plane = self.in_channels
        output_length = self.batch_size
        index_length = self.in_channels
        _col = 1
        for k in self.kernel_size:
            n_output_plane *= k
        for c in self._col:
            output_length *= c
            index_length *= c
            _col *= c

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

        if self.vol is None:
            self.vol = zeros([n_output_plane, output_length])

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
                                if data_col_idx < x.size and data_vol_idx < self.vol.size:
                                    self.vol[int(data_vol_idx)] += x[int(data_col_idx)]


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
        super(MaxPool2d, self).__init__(2   , kernel_size, stride, padding, dilation, return_indices, ceil_mode)


class MaxPool3d(_MaxPoolNd):
    kernel_size: Union[int, List[int]]
    stride: Union[int, List[int]]
    padding: Union[int, List[int]]
    dilation: Union[int, List[int]]

    def __init__(self, kernel_size: Union[int, List[int]], stride: Optional[Union[int, List[int]]] = None,
                 padding: Union[int, List[int]] = 0, dilation: Union[int, List[int]] = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool3d, self).__init__(3, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
