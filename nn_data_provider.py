from DataSet import DataSet
import torch
import json


batch_size = 12
data_size = 10
iters = 1000
shape_x = 224
shape_y = 224


ds = DataSet('data/HFI_SkyMap_100_2048_R2.02_full.fits', batch_size=batch_size)

def normalize_batch(x, y):
    # usage StandardScaler
    s = 0.00012398179470333063
    u = 4.6874553e-05
    return (x - u) / s, (y - u) / s

def get_data(i):
    tx, ty = ds.get_batch(smooth_deg=0.15, with_samples=False, shape_x=shape_x, shape_y=shape_y)
    tx, ty = normalize_batch(tx, ty)
    return tx.tolist(), ty.tolist()

for i in range(0, iters):
    print(f'{i}/{iters}', end='\r')
    xs, ys = [], []
    for j in range(data_size):
        x, y = get_data(j)
        xs.append(x)
        ys.append(y)

    with open(f'train/x_{i}.txt', 'w+') as f:
        f.write(json.dumps(xs))
    with open(f'train/y_{i}.txt', 'w+') as f:
        f.write(json.dumps(ys))
