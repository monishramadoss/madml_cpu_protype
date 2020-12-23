from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
from typing import List, Union
import numpy as np

# from .nn.module import module_cache, execution_order
_gradients = {}
_tensors = []


def _convert_to_float(size: int, arr: List[bytes]) -> List[float]:
    ret_data = []
    ret_data.extend([bytearray(arr[i:i + 4]) for i in range(0, size, 4)])
    for i in range(len(ret_data)):
        ret_data[i] = struct.unpack("f", ret_data[i])
    return ret_data


class tensor(object):
    shape: List[int]
    host_data: List[Union[float, int, bytes, bool]]
    on_device: bool

    def __init__(self, data: Union[List[Union[float, int, bytes, bool]], np.ndarray], shape: List[int] = []) -> None:
        if isinstance(data, np.ndarray):
            self.host_data = data.reshape(-1).tolist()
            self.shape = data.shape
        else:
            self.host_data = data
            self.shape = shape

        assert (len(self.shape) > 0)
        self.size = 1
        for s in self.shape:
            self.size *= s
        assert len(self.host_data) == self.size
        _tensors.append(id(self))
        self.on_device = False
        self.parent = []
        self.children = []

    def __len__(self):
        return self.shape[0]

    def T(self):
        assert len(self.shape) == 2
        self.shape = [self.shape[1], self.shape[0]]
        return self

    def numpy(self):
        return np.array(self.host_data).reshape(self.shape)

    def __getitem__(self, idx: int):
        assert (self.size > idx)
        new_shape = self.shape[1:]
        new_size = 1
        for s in new_shape:
            new_size *= s
        new_data = [self.host_data[idx * new_size + i] for i in range(new_size)]
        return tensor(new_data, new_shape)

    def __setitem__(self, key: int, value) -> None:
        assert (self.size > key)
        assert (type(value) == type(self))
        for i in range(value.size):
            self.host_data[key * value.size + i] = value.host_data[i]

    def copy(self):
        return tensor(self.host_data, self.shape)

    def reshape(self, shape: List[int]):
        _size = 1
        for s in shape:
            _size *= s
        if _size < 0:
            s = self.size // _size
            shape[shape.index(-1)] = abs(s)
            _size *= s
        _size = abs(_size)
        assert (_size == self.size)
        self.shape = shape


    def link(self, t) -> None:
        _gradients[id(self)] = id(t)

    @property
    def gradient(self):
        if id(self) not in _gradients.keys():
            _gradients[id(self)] = tensor([0 for _ in range(self.size)], self.shape)
        return _gradients[id(self)]

    @property
    def grad_data(self):
        if id(self) not in _gradients.keys():
            _gradients[id(self)] = tensor([0 for _ in range(self.size)], self.shape)
        return _gradients[id(self)]

    def backward(self):
        for x in reversed(self.parent):
            if not x.visited[id(self)]:
                print(type(x), 'backward')
                y = x.backward()
                x.visited[id(self)] = True
                if isinstance(y, tensor):
                    y.backward()
        return

    def numpy(self):
        return np.array(self.host_data).reshape(self.shape)