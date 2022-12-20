from DataSet import DataSet
import numpy as np
import logging


logging.basicConfig(filename='data_provider.log', filemode='w', level=logging.WARN)
batch_size = 1024
iters = 2000
shape_x = 128
shape_y = 128

files = [
    'data/LFI_SkyMap_070_2048_R2.01_full.fits',
    'data/HFI_SkyMap_100_2048_R2.02_full.fits',
    'data/HFI_SkyMap_143_2048_R2.02_full.fits',
    'data/HFI_SkyMap_217_2048_R2.02_full.fits',
    'data/HFI_SkyMap_353_2048_R2.02_full.fits',
    'data/HFI_SkyMap_545_2048_R2.02_full.fits',
    'data/HFI_SkyMap_857_2048_R2.02_full.fits'
]
ds = DataSet(files, batch_size=batch_size)

def normalize_batch(xy, i):
    # standart scaler
    xy[:, 0] = (xy[:, 0] - 4.3770884e-05) / 0.0001558874
    xy[:, 1] = (xy[:, 1] - 4.2957174e-05) / 0.00012528372
    xy[:, 2] = (xy[:, 2] - 7.015532e-05) / 0.00011884114

    # xy[:, 3] = (xy[:, 3] - 0.00019576303) / 0.00028319965
    # log-standart scaler
    xy[:, 3] = (np.log(xy[:, 3] + abs(-0.0004694812) + 1e-9) - -7.313942) / 0.33913994

    # log-min-max scaler with [-2, 2] interval
    xy[:, 4] = (np.log(xy[:, 4] + abs(-0.00018964667) + 1e-9) - -7.2715349197387695) / (-5.037784576416016 - -7.2715349197387695) * 4 - 2
    xy[:, 5] = (np.log(xy[:, 5]) - -0.8962784516811371) / (2.3570188379287718 - -0.8962784516811371) * 4 - 2
    xy[:, 6] = (np.log(xy[:, 6]) - -0.17313669502735138) / (3.4202518463134766 - -0.17313669502735138) * 4 - 2

def get_data(i):
    tmp, txy, sx, sxy = ds.get_batch(smooth_deg=0.15, with_samples=True, shape_x=shape_x, shape_y=shape_y)
    normalize_batch(txy, i)
    return txy, sxy

for i in range(0, iters):
    print(f'{i}/{iters}', end='\r')
    xy, sxy = get_data(i)
    np.savez(f'train/xy_{i}.npz', data=np.array(xy, dtype=np.float32), head=np.array(sxy))
