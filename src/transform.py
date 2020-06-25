import math
import numpy as np
import cv2 

class Transformer(object):
    def __init__(self, perspective):
        self.perspective = np.eye(3)
        if perspective is not None:
            if type(perspective) is list:
                for t in reversed(perspective):
                    self.perspective = np.dot(self.perspective, t.perspective)
            elif isinstance(perspective, Transformer):
                self.perspective = perspective.perspective
            else:
                raise ValueError
                
        

    def transform_points(self, xx1,yy1):
        original_shape = xx1.shape
        xx2 = xx1.reshape(-1)
        yy2 = yy1.reshape(-1)
        points = np.stack([xx2,yy2,np.ones(len(yy2))])
        rs = np.dot(self.perspective, points)
        xx3 = rs
        return rs[0].reshape(original_shape), rs[1].reshape(original_shape)

    def transform_image(self, image):
        rs = cv2.warpPerspective(image, self.perspective, (image.shape[1], image.shape[0]))
        return rs

    def transform_array(self, array):
        array[:,:,[0,1]] = array[:,:,[1,0]]
        a = np.concatenate([array, np.ones(array.shape[:2] + (1,))], axis=2)
        a = np.expand_dims(a, 3)
        rs = np.matmul(self.perspective, a)
        rs = rs[:,:,:2,0]
        rs[:,:,[0,1]] = rs[:,:,[1,0]]
        return rs
    
    def transform_field(self, field): #NM2
        head = np.moveaxis(np.mgrid[0:field.shape[0]:1, 0:field.shape[1]:1], 0, 2)
        tail = head + field
        h1 = self.transform_array(head)
        t1 = self.transform_array(tail)
        rs = cv2.warpPerspective((t1 - h1), self.perspective, (field.shape[1], field.shape[0]))
        return rs


class Combined(Transformer):
    def __init__(self, ts):
        self.perspective = np.eye(3)
        for t in reversed(ts):
            self.perspective = np.dot(self.perspective, t.perspective)

class RotationScale(Transformer):
    def __init__(self, origin, angle, scale=1.0):
        self.perspective = cv2.getRotationMatrix2D(origin, angle, scale)
        self.perspective = np.vstack([self.perspective, [0,0,1]])
        
class Translation(Transformer):
    def __init__(self, dx, dy):
        self.perspective = np.array([[1,0,dx],[0,1,dy],[0,0,1]])

class Perspective(Transformer):
    def __init__(self, src, dst):
        self.perspective = cv2.getPerspectiveTransform(src, dst)


# def transfortiveTransform(src, dst)

# def transform(yy, xx, ts):
#     combined = np.eye(3)
#     for t in reversed(ts):
#         combined = np.dot(combined, t.perspective)
#     points = np.stack([xx1,yy1,np.ones(len(yy1))])
#     rs = np.dot(combined, points)
#     return rs[1], rs[0]


def change_perspective_corners(shape, tl, tr, br,  bl, range_width=(0.06,0.06), method='uniform'):
    height = shape[0]
    width = shape[1]
    inp = np.array([tl, tr, br,  bl])
    ds_x = np.random.normal(0, range_width[0], size=4)
    ds_y = np.random.normal(0, range_width[1], size=4)
    inp[:,0] += ds_x
    inp[:,1] += ds_y
    inp[:,0] *= width
    inp[:,1] *= height
    inp = inp.astype(int)
    np.clip(inp[:,0], 0, width, out=inp[:,0])
    np.clip(inp[:,1], 0, height, out=inp[:,1])
    return inp.astype(np.float32)


if __name__ == '__main__':
    # import os, sys
    # sys.path.append('/home/loitg/workspace/backbone/src/')
    from loi import createStandardGrid
    from khoi import ppp

    image = cv2.imread('./test_data/sample.jpg')
    yy1, xx1 = createStandardGrid(image.shape, 10)
    t0 = RotationScale((image.shape[1]/2, image.shape[0]/2), 0, 0.8)
    t1 = RotationScale((image.shape[1]/2, image.shape[0]/2), 45, 1.0)
    t2 = Translation(50,50)
    height, width = image.shape[:2]
    src = np.array([[0,0], [width,0], [width, height], [0, height]], dtype=np.float32)
    dst = change_perspective_corners(image.shape, (0.1,0.1), (0.9,0.1), (0.9,0.9), (0.1,0.9))
    t3 = Perspective(src, dst)
    ts = [t1,t2,t3,t0]

    Ta = Transformer([t1,t2,t3,t0])
    Tb = Transformer(t3)

    xx1,yy1 = Ta.transform_points(xx1,yy1)
    ppp(xx1, yy1, '')

    new_image = Ta.transform_image(image)
    cv2.imwrite('./test_data/output.jpg', new_image)