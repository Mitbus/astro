from DataSet import DataSet
import numpy as np


# config

batch_size = 1024
iters = 1000
shape_x = 128
shape_y = 128


ds = DataSet(['data/HFI_SkyMap_100_2048_R2.02_full.fits'], batch_size=batch_size)
s = np.array([np.std(d.wmap) for d in ds.m])
u = np.array([np.mean(d.wmap) for d in ds.m])

# TODO: move to DataMap
def normalize_batch(x, y, s = 0.00012398179470333063,  u = 4.6874553e-05):
    # usage StandardScaler
    # Reshape s, u with constants for each DataSet
    if type(u) != float:
        u = u.reshape([7,1,1])
    if type(s) != float:
        s = s.reshape([7,1,1])
    return (x - u) / s, (y - u) / s

def get_data():
    tx, ty = ds.get_batch(smooth_deg=0.15, with_samples=False, shape_x=shape_x, shape_y=shape_y)
    tx, ty = normalize_batch(tx, ty, s=s, u=u)
    return tx, ty

for i in range(0, iters):
    print(f'{i}/{iters}', end='\r')
    xs, ys = get_data()

    np.save(f'train/x_{i}', np.array(xs, dtype=np.float32))
    np.save(f'train/y_{i}', np.array(ys, dtype=np.float32))
