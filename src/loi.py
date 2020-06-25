import cv2
import numpy as np
import itertools
from scipy.interpolate import LinearNDInterpolator
from PIL import Image
from matplotlib import pyplot as plt

import os, sys
sys.path.append('/home/loitg/workspace/backbone/src/')
from utils import Coord, Line
from khoi import  ppp, transform

import math
import numpy as np

def createStandardGrid(shape, cell_size=10, start=(0,0)):
    yrange = range(start[0], shape[0], cell_size)
    if yrange[-1] < shape[0]-1:
        yrange = list(yrange)
        yrange.append(shape[0]-1)
    xrange = range(start[1], shape[1], cell_size)
    if xrange[-1] < shape[1]-1:
        xrange = list(xrange)
        xrange.append(shape[1]-1)
    a, b = np.meshgrid(xrange, yrange)
    return a, b
    
def multiple_transform(image, xx,yy,infos):
    for xr, yr, angle, alpha, method in infos:
        p0 = Coord(int(xr*image.shape[1]), int(yr*image.shape[0]))
        angle = angle/180.0*math.pi
        max_dim_size = max(image.shape[1], image.shape[0])
        v = Coord(math.cos(angle)*max_dim_size/7 , math.sin(angle)*max_dim_size/7 )
        line = Line(p0, p0+v)        
        xx, yy = transform(xx, yy, max_dim_size, v, line, alpha, method=method)
    return xx, yy
   
def centralize(xs ,ys):
    xs -= min(xs)
    ys -= min(ys)
    return xs, ys


import numpy as np
def alignGuidelines(gls):
    # rotate matrix
    sumv = Coord(0,0)
    for gl in gls:
        sumv += gl[-1] - gl[0]
    l = sumv.norm()
    cosa = sumv.x/l
    sina = sumv.y/l
    mat2points = np.array([[cosa, sina],[-sina, cosa]])
    points2mat = np.array([[cosa, -sina],[sina, cosa]])
    
    a_gls = []
    for gl in gls:
        gl = np.array([[p.x, p.y] for p in gl])
        gl_rotated = np.dot(gl, points2mat) #[point* mat2points for point in gl]
        gl_rotated[:,1] = gl_rotated.mean(axis=0)[1]
        gl_rotate_back = np.dot(gl_rotated, mat2points)
        a_gls.append([Coord(row[0], row[1]) for row in gl_rotate_back])
    return a_gls
    

def gls2arr(gls):
    a = [item for sublist in gls for item in sublist]
    return np.array([[b.y, b.x] for b in a])

def arr1gls(points, grid_shape):
    return arr2gls(points[:,0], points[:,1], grid_shape)

def arr2gls(yy1, xx1, grid_shape):
    np_gls = np.stack([yy1.reshape(grid_shape), xx1.reshape(grid_shape)], axis=2)
    gls = []
    for np_gl in np_gls:
        a = [Coord(p[1],p[0]) for p in np_gl]
        gls.append(a)
    return gls

def arr2gls2(yy1, xx1, grid_shape):
    np_gls = np.stack([yy1.reshape(grid_shape), xx1.reshape(grid_shape)], axis=2)
    gls = []
    for np_gl in np_gls:
        a = [Coord(p[1],p[0]) for p in np_gl]
        gls.append(a)
    return gls

# MLS Interpoator
def mls(control_points, dst_points, points):
    w = []
    for cp in control_points:
        a = cp - points
        w.append(1.0/np.sum(a**2, axis=1))
        
    tuso = []
    for i in range(len(w)):
        tuso.append(w[i][:,np.newaxis] * control_points[i])
    pstar = sum(tuso)/sum(w)[:,np.newaxis]
    
    tuso = []
    for i in range(len(w)):
        tuso.append(w[i][:,np.newaxis] * dst_points[i])
    qstar = sum(tuso)/sum(w)[:,np.newaxis]
    
    cp_hat = []
    for cp in control_points:
        cp_hat.append(cp - pstar)

    cq_hat = []
    for cq in dst_points:
        cq_hat.append(cq - qstar)
        
    B = []
    for i, cph in enumerate(cp_hat):
        a = np.expand_dims(cp_hat[i], 2)
        b = np.expand_dims(cp_hat[i], 1)
        B.append(np.matmul(a,b)*np.expand_dims(w[i],(1,2)))
    B = np.linalg.inv(sum(B))
    
    temp = np.matmul(np.expand_dims(points - pstar,1), B)
    A = []
    f = qstar
    for i, cqh in enumerate(cq_hat):
        Aj = np.matmul(temp, np.expand_dims(cp_hat[i]*w[i][:,np.newaxis], 2))
        f += (Aj).squeeze(1) * cqh

    return f


def drawField(image, field, mask=None): #(NM2)
    image_field = image.copy()
    for i in range(0, image.shape[0], 10):
        for j in range(0, image.shape[1], 10):
            v = field[i,j]
            image_field = cv2.line(image_field, (j,i), (j+int(v[1]), i + int(v[0])), (255,255,255), 1)
    return image_field

def reconstructFromField(image, deviation_field):
    grid = np.moveaxis(np.mgrid[0:image.shape[0]:1, 0:image.shape[1]:1], 0, 2)
    new_grid = (grid + deviation_field).astype(int)
    np.clip(new_grid[:,:,0], 0,image.shape[0]-1, out=new_grid[:,:,0])
    np.clip(new_grid[:,:,1], 0,image.shape[1]-1, out=new_grid[:,:,1])
    new_image = np.ndarray(shape=image.shape, dtype=np.uint8)
    new_image[new_grid[:,:,0].reshape(-1), new_grid[:,:,1].reshape(-1)] = image[grid[:,:,0].reshape(-1), grid[:,:,1].reshape(-1)]
    return new_image

class GridMap(object):

    def __init__(self, srcPoints, dstPoints):
        self.srcPoints = srcPoints
        self.dstPoints = dstPoints
        self.f = LinearNDInterpolator(srcPoints, dstPoints)
        self.f_inverse = LinearNDInterpolator(dstPoints, srcPoints)

    def transformArrayScalar(self, image): # arr (N,M,k) # keep size
        grid_y, grid_x = np.mgrid[0:image.shape[0]:1, 0:image.shape[1]:1]
        outf = self.f_inverse(grid_y, grid_x)
        outf[outf[:,:,0] >= image.shape[0],0] = np.nan
        outf[outf[:,:,0] < 0,0] = np.nan
        outf[outf[:,:,1] >= image.shape[1],1] = np.nan
        outf[outf[:,:,1] < 0,1] = np.nan
        new_index = np.nan_to_num(outf).astype(int)
        image[0,0]= 0
        newy = new_index.reshape(-1,2)[:,0]
        newx = new_index.reshape(-1,2)[:,1]
        new_image = np.ndarray(shape=image.shape)
        new_image[grid_y.reshape(-1), grid_x.reshape(-1)] = image[newy, newx]
        return np.nan_to_num(new_image)
    
    def _fillIn(self, phannguyen, image, grid_y, grid_x):
        phannguyen[phannguyen[:,:,0] >= image.shape[0],0] = np.nan
        phannguyen[phannguyen[:,:,0] < 0,0] = np.nan
        phannguyen[phannguyen[:,:,1] >= image.shape[1],1] = np.nan
        phannguyen[phannguyen[:,:,1] < 0,1] = np.nan
        new_index = np.nan_to_num(phannguyen).astype(int)
        image[0,0]= 0
        newy = new_index.reshape(-1,2)[:,0]
        newx = new_index.reshape(-1,2)[:,1]
        new_image = np.ndarray(shape=image.shape)
        new_image[grid_y.reshape(-1), grid_x.reshape(-1)] = image[newy, newx]
        return new_image
    
    def transformArrayScalar_bilinear(self, image):
        grid_y, grid_x = np.mgrid[0:image.shape[0]:1, 0:image.shape[1]:1]
        outf = self.f_inverse(grid_y, grid_x)
        phannguyen = np.floor(outf)
        phannguyen1 = phannguyen + np.array([0,1])
        phannguyen2 = phannguyen + np.array([1,1])
        phannguyen3 = phannguyen + np.array([1,0])
        phanle = outf - np.floor(outf)
        p0 = phanle[:,:,0] * phanle[:,:,1]
        p1 = (1-phanle[:,:,1]) * phanle[:,:,0]
        p3 = phanle[:,:,1] * (1-phanle[:,:,0])
        p2 = (1-phanle[:,:,0]) * (1-phanle[:,:,1])
        img0 = self._fillIn(phannguyen, image, grid_y, grid_x)
        img1 = self._fillIn(phannguyen1, image, grid_y, grid_x)
        img2 = self._fillIn(phannguyen2, image, grid_y, grid_x)
        img3 = self._fillIn(phannguyen3, image, grid_y, grid_x)
        img = img2*p0[:,:,np.newaxis] + img3*p1[:,:,np.newaxis] + img0*p2[:,:,np.newaxis] + img1*p3[:,:,np.newaxis]
        return np.nan_to_num(img).astype(np.uint8)

    def transformArrayVector(self, field, mask=None): # arr (N,M,2)
        grid_y, grid_x = np.mgrid[0:field.shape[0]:1, 0:field.shape[1]:1]
        head = self.f(grid_y, grid_x)
        grid_x += field[:,:,1]
        grid_y += field[:,:,0]
        tail = self.f(grid_y, grid_x)
        return self.transformArrayScalar(tail-head)
    
    def deviationFromShape(self, shape):
        grid = np.moveaxis(np.mgrid[0:shape[0]:1, 0:shape[1]:1], 0, 2)
        new_grid = self.f(grid)
        deviation_field = (new_grid - grid) #np.stack([new_grid_y - grid_y, new_grid_x - grid_x], axis=2)
        deviation_field = np.nan_to_num(deviation_field)
        deviation_field -= np.mean(deviation_field, axis=(0,1))
        return deviation_field