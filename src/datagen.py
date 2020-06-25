from numpy import random as rd
import random
from transform import RotationScale, Translation, Transformer, change_perspective_corners, Perspective

# add background ?

def gensmall_matrix(image, yy, xx):
    angle = rd.normal(0,3)
    scale = rd.uniform(0.9,1.0)
    t1 = RotationScale((image.shape[1]/2, image.shape[0]/2), angle, scale) ###
    a = int(0.05*image.shape[1])
    dx, dy = rd.randint(-a,a, size=2)
    t2 = Translation(dx, dy) ###
    ts = random.sample([t1, t2], 2)
    t = Transformer(ts)
    return t, scale
    


def geninside_matrix(image, yy, xx):
    angle = rd.normal(0,30)
    t1 = RotationScale((image.shape[1]/2, image.shape[0]/2), angle, 1.0) ###
    height, width = image.shape[:2]
    src = np.array([[0,0], [width,0], [width, height], [0, height]], dtype=np.float32)
    dst = change_perspective_corners(image.shape, (0.1,0.1), (0.9,0.1), (0.9,0.9), (0.1,0.9))
    t3 = Perspective(src, dst) ###

    return Transformer(random.sample([t1, t2, t3], 3)), scale

def syn_augment_matrix(image, yy, xx):
    if rd.rand() < 1.0:
        return gensmall_matrix(image, yy, xx)
    else:
        return geninside_matrix(image, yy, xx)

def createRandomInfos():
    N = rd.randint(1,6) ### fold/curl counts
    infos = []
    for i in range(N):
        center_r_x = rd.rand()
        center_r_y = rd.rand()
        direction = rd.uniform(360)
        #print('%.2f,%.2f--%.2f--%.2f' % (center_r_x, center_r_y, direction, strength))
        method = random.choice(['fold', 'curl'])
        if method == 'fold':
            strength = rd.uniform(0.4,1.3)
        elif method == 'curl':
            strength = rd.uniform(2.0,3.0)
        infos.append((center_r_x, center_r_x, direction, strength, method))
    return infos