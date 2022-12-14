import numpy as np
from Config import *

def test(model, data_iter, optimezer, criterion, lr_low=1e-6, lr_max=1e-1, mult=1.1):
    errors = []
    lr = lr_low
    x, y_old = next(data_iter)
    for x,y in data_iter:
        if lr > lr_max:
            return np.array(errors)
        optim = optimezer(lr)
        model.zero_grad()
        y_hat = model(x)

        err = criterion(y_hat, y)
        # y_old = y
        err.backward()
        item = err.item()
        if item > max_test_error:
            if len(errors) == 0:
                item = 0
            else:
                item = errors[-1][1]
        errors.append([lr, item])
        print('lr, error: \t', errors[-1])

        optim.step()
        lr *= mult
