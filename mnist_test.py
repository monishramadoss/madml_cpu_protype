import numpy as np

import madml
import madml.nn as nn
import madml.optimizer as optimizer

np.random.seed(seed=1024)


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
    batchsize = 1000
    epochs = 10

    x, y, x1, y1 = load()
    x = x.reshape((-1, batchsize, 28 * 28))
    y = y.reshape((-1, batchsize, 1))

    t_x = madml.tensor(x / 225.)
    t_y = madml.tensor(y).onehot(label_count=10)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss(with_logit=True)
    optim = optimizer.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for i in range(x.shape[0]):
            optim.zero_grad()
            logit = model(t_x[i])
            loss = loss_fn(logit, t_y[i])
            loss.backward()
            optim.step()
            print('===', i, logit.shape, loss.host_data, loss_fn.accuracy())
            if i % 10 == 0:
                print('logit [', end=' ')
                for j in range(10):
                    print(logit.host_data[0][j], end='] ' if j == 9 else ', ')
                print(': target [', end=' ')
                for j in range(10):
                    print(t_y[i].host_data[0][j], end=']\n' if j == 9 else ', ')


def identity_train_loop(model_class=identity_model):
    model = model_class()
    x = np.ones((2, 32))
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
    print(logit.host_data)


if __name__ == '__main__':
    dnn_train_loop()
