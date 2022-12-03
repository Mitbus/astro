from DataMap import DataMap, Sample
import random
from matplotlib import pyplot as plt
import numpy as np
import torch

NOT_SET = -1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSet:
    def __init__(self, path, batch_size=16):
        self.m = DataMap(path)
        self.wmap_min = self.m.wmap.min()
        self.wmap_max = self.m.wmap.max()
        self.m.set_line(NOT_SET, width=5)
        self.batch_size = batch_size
            
    def x_proj(self, sample, shape=300):
        return self.m.get_proj(sample=sample,shape=shape)
    def y_proj(self, sample, smooth_deg=None, shape=100):
        if smooth_deg is None:
            return self.m.get_proj(sample=sample,shape=shape)
        else:
            return self.m.get_proj(sample=sample,shape=shape, smooth=True, smooth_deg=smooth_deg)
    
    def get_item(self, smooth_deg=None, with_samples=False, shape_x=300, shape_y=100):
        """returns X, y"""
        phi_deg = random.random() * 360
        theta_deg = (random.random() - 0.5) * 180
        sx = Sample(2, phi_deg, theta_deg)
        sy = Sample(1, phi_deg, theta_deg)
        x = self.x_proj(sx, shape=shape_x)
        y = self.y_proj(sy, smooth_deg=smooth_deg, shape=shape_y)
        if np.all(x != NOT_SET):
            if with_samples:
                return [x], [y], sx, sy
            else:
                return [x], [y]
        else:
            return self.get_item(smooth_deg=smooth_deg, with_samples=with_samples, shape_x=shape_x, shape_y=shape_y)
    def set_batch_size(self, batch):
        self.batch_size = batch
    def get_batch(self, smooth_deg=None, with_samples=False, shape_x=300, shape_y=100):
        xs, ys = [], []
        sx, sy = [], []
        for i in range(self.batch_size):
            tmp = self.get_item(smooth_deg=smooth_deg, with_samples=with_samples, shape_x=shape_x, shape_y=shape_y)
            if tmp is not None:
                xs.append(tmp[0])
                ys.append(tmp[1])
                if with_samples:
                    sx.append(tmp[2])
                    sy.append(tmp[3])
        if len(xs) == 0:
            return None
        if with_samples:
            return np.array(xs), np.array(ys), sx, sy
        else:
            return np.array(xs), np.array(ys)
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return self
    def __next__(self):
        batch = self.get_batch()
        if batch is None:
            raise StopIteration()
        return batch