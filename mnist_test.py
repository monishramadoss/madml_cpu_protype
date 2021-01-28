import gzip
import os
import pickle
from urllib import request

import numpy as np

import madml
import madml.nn as nn
import madml.optimizer as optimizer

np.random.seed(seed=1024)

filename = [["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"]]

if not os.path.exists('./data'):
    os.makedirs('./data')


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], './data/' + name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open('./data/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open('./data/' + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("./data/mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    if not os.path.exists('./data/mnist.pkl'):
        download_mnist()
        save_mnist()


def load():
    init()
    with open("./data/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


class cnn_mnist_model(nn.Module):
    def __init__(self):
        super(cnn_mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 48, 3)
        self.fc1 = nn.Linear(48 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # 32 x 28 x 28
        x = self.relu1(x)
        x = self.pool(x)  # 32 x 14 x 14
        x = self.conv2(x)  # 46 x 12 x 12
        x = self.relu2(x)
        x.flatten()
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


class dnn_mnist_model(nn.Module):
    def __init__(self):
        super(dnn_mnist_model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        return x


class identity_model(nn.Module):
    def __init__(self):
        super(identity_model, self).__init__()
        self.fc1 = nn.Linear(32, 32)

    def forward(self, x):
        x = self.fc1(x)
        return x


def cnn_train_loop(model_class=cnn_mnist_model):
    model = model_class()
    batchsize = 16
    x, y, x1, y1 = load()
    x = x.reshape((-1, 1, 28, 28))
    y = y.reshape((-1, 1))

    t_x = madml.tensor(x)
    t_y = madml.tensor(y).onehot()
    loss_fn = nn.MSELoss()
    optim = optimizer.Adam(model.parameters(), lr=1e-2)
    for i in range(10):
        optim.zero_grad()
        logit = model(t_x)
        loss = loss_fn(logit, t_y)
        loss.backward()
        optim.step()
        exit_statement = ((loss.host_data < .4).all() or
                          (np.abs(loss.host_data) == np.inf) or
                          (loss.host_data == np.nan)) and (i != 0)

        if exit_statement:
            for n in range(batchsize):
                print('logit', np.argmax(logit.host_data[n]), end=': ')
                print('target', np.argmax(t_y.host_data[n]))
            break
        else:
            print('===', i, logit.shape, loss.host_data, loss.accuracy())


def dnn_train_loop(model_class=dnn_mnist_model):
    model = model_class()
    batchsize = 100
    epochs = 10

    x, y, x1, y1 = load()
    x = x.reshape((-1, batchsize, 28 * 28))
    y = y.reshape((-1, batchsize, 1))

    t_x = madml.tensor(x / 1.)
    t_y = madml.tensor(y).onehot(label_count=10)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss(with_logits=True)
    optim = optimizer.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        for i in range(x.shape[0]):
            optim.zero_grad()
            logit = model(t_x[i])
            loss = loss_fn(logit, t_y[i])
            loss.backward()
            optim.step()
            print('===', i, logit.shape, loss.host_data, loss_fn.accuracy())


def identity_train_loop(model_class=identity_model):
    model = model_class()
    x = np.random.randn(2, 32)
    t_x = madml.tensor(x)
    t_y = madml.tensor(x.copy())
    loss_fn = nn.MSELoss()
    optim = optimizer.Adam(model.parameters(), lr=1e-2)
    logit = None

    for _ in range(100):
        optim.zero_grad()
        logit = model(t_x)
        loss = loss_fn(logit, t_y)
        loss.backward()
        optim.step()
        print(logit.shape, loss.host_data, loss_fn.accuracy())
    print()


if __name__ == '__main__':
    identity_train_loop()
