from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import List

from madml import tensor, zeros
from .module import Module


class flatten(Module):
    old_shape: List[int]

    def __init__(self) -> None:
        super(flatten, self).__init__()
        self.old_shape = []

    def forward_cpu(self, x: tensor) -> tensor:
        self.old_shape = x.shape
        size = 1
        for s in x.shape[1:]:
            size *= s
        x.reshape([x.shape[0], size])
        self.cache.append(x)
        return x

    def backward_cpu(self) -> tensor:
        x = self.cache[0]
        x.reshape(self.old_shape)
        x.gradient.reshape(self.old_shape)
        return x


class transpose(Module):
    __constants__ = ['axes']
    axes: List[int]
    stride: List[int]
    old_shape: List[int]
    new_shape: List[int]

    def __init__(self, axes: List[int]) -> None:
        super(transpose, self).__init__()
        self.axes = axes
        self.old_shape = []
        self.new_shape = []
        self.stride = [1 for _ in range(len(axes) * 3)]

    def forward_cpu(self, x: tensor) -> tensor:
        assert (len(x.shape) == len(self.axes))
        if self.old_shape == [] or self.new_shape == []:
            self.old_shape = x.shape
            self.new_shape = [self.old_shape[self.axes[i]] for i in range(len(self.axes))]
            self.stride = self.prepare_shape(self.old_shape, self.new_shape)

        y = zeros(self.new_shape)
        for i in range(x.size):
            old_pos = 0
            new_pos = i
            for j in range(len(x.shape)):
                order = self.stride[j]
                old_pos += (new_pos / self.stride[len(x.shape) + j] * self.stride[len(x.shape) * 2 + order])
                new_pos %= self.stride[len(x.shape) + j]

            y.host_data[i] = x.host_data[old_pos]
        self.cache.append(x)
        self.cache.append(y)
        return y

    def backward_cpu(self) -> None:
        x, y = self.cache

        self.prepare_stride(self.new_shape, self.old_shape)
        for i in range(x.size):
            old_pos = 0
            new_pos = i
            for j in range(len(x.shape)):
                order = self.stride[j]
                old_pos += (new_pos / self.stride[len(x.shape) + j] * self.stride[len(x.shape) * 2 + order])
                new_pos %= self.stride[len(x.shape) + j]
            x.gradient.host_data[i] = y.gradient.host_data[old_pos]
        return x

    def prepare_stride(self, shape_before: List[int], shape_after: List[int]):
        dims = len(self.axes)
        self.stride[2 * dims - 1] = 1
        self.stride[3 * dims - 1] = 1
        for i in range(dims - 2, 0, -1):
            self.stride[dims * 2 + i] = self.stride[dims * 2 + i + 1] * shape_before[i + 1]
            self.stride[dims + i] = self.stride[dims + i + 1] * shape_after[i + 1]
