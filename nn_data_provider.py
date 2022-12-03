from DataSet import DataSet
import numpy as np


batch_size = 1024
iters = 2000
shape_x = 128
shape_y = 128


ds = DataSet('data/HFI_SkyMap_100_2048_R2.02_full.fits', batch_size=batch_size)

def normalize_batch(xy):
    # usage StandardScaler
    # TODO:  find s, u for each map
    s = 0.00012398179470333063
    u = 4.6874553e-05
    return (xy - u) / s

def get_data():
    txy = ds.get_batch(smooth_deg=0.15, with_samples=False, shape_x=shape_x, shape_y=shape_y)[1]
    txy = normalize_batch(txy)
    return txy

for i in range(0, iters):
    print(f'{i}/{iters}', end='\r')
    xy = get_data()

    np.save(f'train/xy_{i}', np.array(xy, dtype=np.float32))
