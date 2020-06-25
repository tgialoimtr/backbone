class Coord(object):
    ''' Point '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def setT(self, t):
        self.x = t[0]
        self.y = t[1]

    def getT(self):
        return (self.x, self.y)

    t = property(getT, setT)

    def getTInteger(self):
        return (int(self.x), int(self.y))

    ti = property(getTInteger, setT)

    def __str__(self):
        return "({0},{1})".format(self.x,self.y)

    def __add__(self, o):
        return Coord(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Coord(self.x - o.x, self.y - o.y)

    def __pow__(self, o):
        return self.x * o.x + self.y * o.y

    def __truediv__(self, num):
        return Coord(self.x/num, self.y/num)

    def __mul__(self, num):
        return Coord(self.x*num, self.y*num)

    def __copy__(self):
        return Coord(self.x, self.y)
    
    def __deepcopy__(self, memo):
        return Coord(self.x, self.y)

    @staticmethod
    def distance(o1, o):
        return (o1 - o).norm()

    @staticmethod
    def angle(o1, o2):
        ''' Goc xoay quay goc toa do Oxy '''
        a = math.atan2(o2.y, o2.x) - math.atan2(o1.y, o1.x)
        while a > math.pi:
            a -= 2*math.pi
        while a < -math.pi:
            a += 2*math.pi
        return a


    @staticmethod
    def abs_angle(o1, o):
        cosa = (o1**o)/(o1.norm()*o.norm())
        return math.degrees(math.acos(cosa*0.9999))

    
    def rotateAround(self, origin, angle):
        ''' Xoay diem hien tai quanh diem `origin` mot goc `angle` '''
        if type(origin) is Coord:
            o = origin.t
        else:
            o = origin
        dx = self.x - o[0]
        dy = self.y - o[1]
        newdx = math.cos(angle) * dx - math.sin(angle) * dy
        newdy = math.sin(angle) * dx + math.cos(angle) * dy
        self.x = newdx + o[0]
        self.y = newdy + o[1]
        return self

    def norm(self):
        return math.sqrt(self.x*self.x + self.y*self.y)


def rotatePointsAround(origin, points, alpha):
    pass

import math
import numpy as np

class Line(object):
    ''' Line '''
    
    def __init__(self, p1, p2):
        p1_ = np.array([p1.x, p1.y, 1])
        p2_ = np.array([p2.x, p2.y, 1])
        self.line_ = np.cross(p1_, p2_)
    
    def intersect(self, other_line):
        ret_ = np.cross(self.line_, other_line.line_)
        if ret_[2] == 0:
            return None
        else:
            ret_ /= ret_[2]
            return ret_[0], ret_[1]
    
    def distance2point(self, point):
        point_ = np.array([point.x, point.y, 1])
        ret = np.dot(point_, self.line_)
        a = self.line_[0]
        b = self.line_[1]
        return abs(ret/math.sqrt(a*a + b*b))

    def distancebatch(self, x, y):
        assert x.shape[0] == y.shape[0]
        #print(x.shape)
        #print(y.shape)
        point_ = np.stack([x, y, np.ones((x.shape[0]))], axis=1)
        ret = np.dot(point_, self.line_)
        a = self.line_[0]
        b = self.line_[1]
        return abs(ret/math.sqrt(a*a + b*b))