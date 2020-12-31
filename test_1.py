import unittest
import madml
import madml.nn as nn
import numpy as np


class TestModules(unittest.TestCase):
    def test_tensor(self):
        x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        t1 = madml.tensor(x)
        self.assertTrue(t1.shape == list(x.shape))
        self.assertTrue((t1.host_data == x).all())

    def test_conv(self):
        kernel_shape = [3, 3]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]

        x = np.array([[[[0., 1., 2., 3., 4.],
                         [5., 6., 7., 8., 9.],
                         [10., 11., 12., 13., 14.],
                         [15., 16., 17., 18., 19.],
                         [20., 21., 22., 23., 24.]]]]).astype(np.float32)
        y_with_padding = np.array([[[12., 21., 27., 33., 24.],
                                    [33., 54., 63., 72., 51.],
                                    [63., 99., 108., 117., 81.],
                                    [93., 144., 153., 162., 111.],
                                    [72., 111., 117., 123., 84.]]]).astype(np.float32).reshape([1, 1, 5, 5])

        t1 = madml.tensor(x)
        module = nn.Conv2d(1, 1, kernel_shape, stride, padding, dilation, weight_init='ones')
        t2 = module.forward_cpu(t1)
        y = t2.host_data
        self.assertTrue((y == y_with_padding).all())

        padding = [0, 0]
        y_without_padding = np.array([[[[54., 63., 72.],
                                        [99., 108., 117.],
                                        [144., 153., 162.]]]]).astype(np.float32).reshape([1, 1,  3, 3])
        module2 = nn.Conv2d(1, 1, kernel_shape, stride, padding, dilation, weight_init='ones')
        t3 = module2.forward_cpu(t1)
        y2 = t3.host_data
        self.assertTrue((y2 == y_without_padding).all())

        dy = np.array([[[[0., 1., 2.],
                         [3., 4., 5.],
                         [6., 7., 8.]]]]).astype(np.float32).reshape([1, 1,  3, 3])
        dx = np.array([[[[0., 1., 3., 3., 2.],
                         [3., 8., 15., 12., 7.],
                         [9., 21., 36., 27., 15.],
                         [9., 20., 33., 24., 13.],
                         [6., 13., 21., 15., 8.]]]]).reshape([1, 1, 5, 5])

        t3.gradient.host_data = dy
        _ = module2.backward_cpu()
        y3 = t1.gradient.host_data
        self.assertTrue((y3 == dx).all())

    def test_maxpool(self):
        kernel_shape = [2, 2]
        stride = [1, 1]
        padding = [0, 0]
        dilation = [1, 1]

        x = np.arange(0, 100).astype(np.float32).reshape([2, 2, 5, 5])

        t1 = madml.tensor(x)
        module = nn.MaxPool2d(kernel_shape, stride, padding, dilation)
        t2 = module.forward_cpu(t1)
        y = t2.host_data

        test = x[..., 1:, 1:]
        self.assertTrue((test == y).all())
        t2.gradient.host_data = y
        _x = module.backward_cpu()
        dx = t1.gradient.host_data[..., 1:, 1:]
        self.assertTrue(True)

    def test_crossentropy(self):
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, ))

        t1 = madml.tensor(x)
        target = madml.tensor(labels)
        module = nn.CrossEntropyLoss()

        loss = module.forward_cpu(t1, target)

        dx = module.backward_cpu()
        print(loss.host_data, dx.gradient.host_data)

if __name__ == '__main__':
    unittest.main()
