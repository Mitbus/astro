from DataMap import DataMap, Sample
import random
from matplotlib import pyplot as plt
import numpy as np

NOT_SET = -1000

class DataSet:
    def __init__(self, paths, batch_size=16, skip_width=10, replace_line=False):
        self.m = [DataMap(path) for path in paths]
        self.skip_width = skip_width
        if replace_line:
            for dm in self.m:
                dm.set_line(NOT_SET, width=self.skip_width )
        self.batch_size = batch_size
            
    def x_proj(self, freq, sample, shape=300):
        return self.m[freq].get_proj(sample=sample,shape=shape)
    def y_proj(self, freq, sample, smooth_deg=None, shape=100):
        if smooth_deg is None:
            return self.m[freq].get_proj(sample=sample,shape=shape)
        else:
            return self.m[freq].get_proj(sample=sample,shape=shape, smooth=True, smooth_deg=smooth_deg)

    def sample_spherical(self, ndim=3):
        vec = np.random.randn(ndim)
        vec /= np.linalg.norm(vec)
        return vec
    def get_angles(self, vec):
        phi = np.arctan2(vec[1], vec[0]) / np.pi * 180
        theta = np.arccos(vec[2]) / np.pi * 180
        return phi, theta - 90
    
    def get_item(self, smooth_deg=None, with_samples=False, shape_x=300, shape_y=100):
        """returns X, y"""
        # phi_deg = random.random() * 360
        # theta_deg = (random.random() - 0.5) * 180
        vec = self.sample_spherical()
        phi_deg, theta_deg = self.get_angles(vec)
        # scale = random.random() * 4 + 1
        scale = 1
        if -self.skip_width  <= theta_deg <= self.skip_width:
            return self.get_item(smooth_deg=smooth_deg, with_samples=with_samples, shape_x=shape_x, shape_y=shape_y)
        sx = Sample(scale, phi_deg, theta_deg)
        sy = Sample(scale, phi_deg, theta_deg)
        xs = [self.x_proj(i, sx, shape=shape_x) for i in range(len(self.m))]
        ys = [self.y_proj(i, sy, smooth_deg=smooth_deg, shape=shape_y) for i in range(len(self.m))]
        if with_samples:
            return xs, ys, sx, sy
        else:
            return xs, ys
            
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