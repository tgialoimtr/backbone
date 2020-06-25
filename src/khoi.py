from utils import Coord, Line
import cv2
import numpy as np
import itertools
from PIL import Image
from matplotlib import pyplot as plt

from scipy.interpolate import LinearNDInterpolator


def transform(x, y, grid_size, v, line, alpha, method='fold'):
    # alpha = 0.4
    dd = []
    d = line.distancebatch(x, y)/(grid_size)

    if method == 'fold':
        w = alpha/(d+alpha)
    elif method == 'curl':
        w = 1 - np.power(d, alpha)
                         
    xs = x + v.x * w
    ys = y + v.y * w
    return xs, ys



def ppp(xs, ys, dest):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(10,10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(xs[::2], ys[::2], s=11, c=xs[::2])
    plt.savefig(dest)

from time import time 

# def Interpolate(standargrid, xs, ys, img, name):
#     assert img.shape[2] == 3
#     # Scaling
#     tt = time()
#     Coords = [(x*(img.shape[0]-1)/100.0,y*(img.shape[1]-1)/100.0) for x,y in standargrid]
#     Coords_target = np.stack([xs,ys],axis=1)
#     Coords_target -= np.min(Coords_target, 0)
#     Coords_target = Coords_target / np.max(Coords_target) * 100 
#     Coords_target = Coords_target * np.array([(img.shape[0]-1), (img.shape[1]-1)]) / 100
#     print('A %f' % (time() - tt))


#     tt= time()
#     # Coordinating interpolate
#     f = LinearNDInterpolator(Coords, Coords_target, fill_value=0)
#     # Mapping coordinate
#     a, b = np.meshgrid(np.arange(img.shape[0]),np.arange(img.shape[1]))
#     grid = np.array([a,b]).swapaxes(0,2).reshape(-1,2)
#     Coord_new = f(grid)
#     print('B %f' % (time() - tt))


#     tt = time()
#     Coord_new = np.round(Coord_new).astype(np.int32)
#     Coord_new -= np.min(Coord_new)

#     x_range, y_range = np.max(Coord_new,0) + 1
#     Image_new = np.zeros((x_range * y_range,3))
#     img1 = img.reshape(-1,3)

#     Coord_new = Coord_new[:,0] * y_range + Coord_new[:,1] 
#     grid = grid[:,0] * img.shape[1] + grid[:,1]

#     Image_new[Coord_new,:] = img1[grid,:]

#     print('C %f' % (time() - tt))
#     return Image_new.reshape(x_range, y_range, 3)




