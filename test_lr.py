import numpy as np

def test(model, data_iter, optimezer, criterion, lr_low=1e-6, lr_max=1e-1, mult=1.1):
    errors = []
    lr = lr_low
    for x,y in data_iter:
        if lr > lr_max:
            return np.array(errors)
        optim = optimezer(lr)
        model.zero_grad()
        y_hat = model(x)

        err = criterion(y_hat, y)
        err.backward()
        errors.append([lr, err.item()])
        print('lr, error: \t', errors[-1])

        optim.step()
        lr *= mult
