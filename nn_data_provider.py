from DataSet import DataSet
import torch
import json
import numpy as np


# config

batch_size = 1024
iters = 1000
shape_x = 256
shape_y = 256


ds = DataSet(['data/HFI_SkyMap_100_2048_R2.02_full.fits'], batch_size=batch_size)

def normalize_batch(x, y):
    # usage StandardScaler
    s = 0.00012398179470333063
    u = 4.6874553e-05
    return (x - u) / s, (y - u) / s

def get_data():
    tx, ty = ds.get_batch(smooth_deg=0.15, with_samples=False, shape_x=shape_x, shape_y=shape_y)
    tx, ty = normalize_batch(tx, ty)
    return tx.tolist(), ty.tolist()

for i in range(0, iters):
    print(f'{i}/{iters}', end='\r')
    xs, ys = get_data()

    np.save(f'train/x_{i}', np.array(xs, dtype=np.float32))
    np.save(f'train/y_{i}', np.array(ys, dtype=np.float32))
