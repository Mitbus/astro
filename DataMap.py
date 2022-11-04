import healpy as hp
import numpy as np
import astropy.units as u
from typing import NamedTuple
from functools import partial

class Sample(NamedTuple):
    radius: float
    phi_deg: float
    theta_deg: float

class DataMap:
    def __init__(self, path):
        self.path = path
        self.wmap = hp.read_map(path)
        self.npix = len(self.wmap)
        self.nside = hp.npix2nside(self.npix)
    def view(self, min=None, max=None, rot=[0,0], smooth=False, smooth_deg=0.5, sample: Sample=None):
        if sample is not None:
            rot = [sample.phi_deg, sample.theta_deg]
        if smooth:
            wmap = self.smooth_wmap(smooth_deg)
        else:
            wmap = self.wmap
        if min == None:
            min = wmap.min()
        if max == None:
            max = wmap.max()
        hp.mollview(
            wmap,
            title=self.path,
            unit="mK",
            norm="hist",
            min=min,
            max=max,
            rot=rot
        )
        hp.graticule()
    def crop(self, shape=[500,300], min=None, max=None, rot=[0,0], smooth=False, smooth_deg=0.5, sample: Sample=None):
        if sample is not None:
            rot = [sample.phi_deg, sample.theta_deg]
        if smooth:
            wmap = self.smooth_wmap(smooth_deg)
        else:
            wmap = self.wmap
        if min == None:
            min = wmap.min()
        if max == None:
            max = wmap.max()
        if type(shape) is int:
            shape = [shape, shape]
        hp.gnomview(
            wmap,
            title=self.path,
            unit="mK",
            norm="hist",
            rot=rot,
            xsize=shape[0],
            ysize=shape[1],
            min=min,
            max=max,
        )
        hp.graticule()
    def smooth_wmap(self, smooth_deg):
        if not hasattr(self, '_smooth'):
            self._smooth = {}
        if smooth_deg not in self._smooth:
            self._smooth[smooth_deg] = hp.smoothing(self.wmap, fwhm=(smooth_deg*u.deg).to_value(u.radian))
        return self._smooth[smooth_deg]
    def smooth_only(self,smooth_deg=0.5):
        self.wmap = self.smooth_wmap(smooth_deg)
    def get_line(self, width=1):
        deg_width = width * u.deg
        plane_pixels = hp.query_strip(self.nside, np.pi/2-deg_width.to_value(u.radian), np.pi/2+deg_width.to_value(u.radian))
        return self.wmap[plane_pixels]
    def set_line(self, value, width=1):
        deg_width = width * u.deg
        plane_pixels = hp.query_strip(self.nside, np.pi/2-deg_width.to_value(u.radian), np.pi/2+deg_width.to_value(u.radian))
        self.wmap[plane_pixels] = value
    def get_disc(self, radius=5, theta=0, phi=0, sample: Sample=None):
        if sample is not None:
            radius = sample.radius
            theta = sample.theta_deg
            phi = sample.phi_deg
        source_vector = hp.ang2vec(theta=((90-theta)*u.deg).to_value(u.radian), phi=((phi)*u.deg).to_value(u.radian))
        plane_pixels = hp.query_disc(self.nside, source_vector, radius=(radius*u.deg).to_value(u.radian))
        return self.wmap[plane_pixels]
    def set_disc(self, value, radius=5, theta=0, phi=0, sample: Sample=None):
        if sample is not None:
            radius = sample.radius
            theta = sample.theta_deg
            phi = sample.phi_deg
        source_vector = hp.ang2vec(theta=((90-theta)*u.deg).to_value(u.radian), phi=(phi*u.deg).to_value(u.radian))
        plane_pixels = hp.query_disc(self.nside, source_vector, radius=(radius*u.deg).to_value(u.radian))
        self.wmap[plane_pixels] = value
    def get_poly(self, vertices=[[0,0,0],[0.1,0.1,0.1],[0.1,0.3,0]]):
        plane_pixels = hp.query_polygon(self.nside, vertices)
        return self.wmap[plane_pixels]
    def set_poly(self, value, vertices=[[0,0,0],[0.1,0.1,0.1],[0.1,0.3,0]]):
        plane_pixels = hp.query_polygon(self.nside, vertices)
        self.wmap[plane_pixels] = value
    def get_proj(self, rot=[0,0], angs=[10,10], sample: Sample=None, shape=[2000,2000], smooth=False, smooth_deg=0.5):
        if sample is not None:
            angs = 2*sample.radius
            rot = [sample.phi_deg, sample.theta_deg]
        if smooth:
            wmap = self.smooth_wmap(smooth_deg)
        else:
            wmap = self.wmap
        if type(shape) is int:
            shape = [shape, shape]
        if type(angs) is int:
            angs = [angs, angs]
        lonra = [-angs[0] / 2, angs[0] / 2]
        latra = [-angs[1] / 2, angs[1] / 2]
        proj = hp.projector.CartesianProj(
            rot=rot, lonra=lonra, latra=latra,
            xsize=shape[0], ysize=shape[1]
        ).projmap(wmap, vec2pix_func=partial(hp.vec2pix, self.nside))
        return proj[::-1].copy()