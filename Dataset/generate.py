import matplotlib.markers
import numpy as np
import random
from scipy.optimize import linprog
import flow_vis
from matplotlib.path import Path
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]

def drawImage(name1, name2, flowname, edgeName, flowColorname=None):
    image = np.zeros(im_size)
    image2 = np.zeros(im_size)
    flow = np.zeros(flow_size, dtype=np.float32)
    edgeField = np.zeros((flow_size[0],flow_size[1]), dtype=np.int8)
    image, image2, flow, edgeField = blank.add_to_image(image, image2, flow, edgeField)
    # imwrite(name1, image.astype(np.uint8))
    image = crop_center(image, im_size[0]-pad, im_size[1]-pad)
    image2 = crop_center(image2, im_size[0] -pad, im_size[1] -pad)
    flow = crop_center(flow, im_size[0] -pad, im_size[1] -pad)
    edgeField = np.repeat(edgeField[:,:,np.newaxis], 3, axis=2) * 255
    edgeField = crop_center(edgeField, im_size[0] -pad, im_size[1] -pad)
    imageio.imwrite(name1, image.astype(np.uint8))
    imageio.imwrite(name2, image2.astype(np.uint8))
    imageio.imwrite(edgeName, edgeField.astype(np.uint8))
    writeFlow(flowname, flow)






def in_hull(points, x):
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


class Image:

    def __init__(self):
        self.w, _, _ = im_size
        self.circle = Circle()
        self.poly = Polygon()

    def add_to_image(self, image, image2, flow, edgeField):
        num_objects = np.random.randint(3, 6)
        l = [self.poly, self.poly, self.poly, self.circle]
        for _ in range(num_objects):
            shape = random.choice(l)
            image, image2, flow, edgeField = shape.add_to_image(image, image2, flow, edgeField)
        return image, image2, flow, edgeField


class Shape:

    def __init__(self):
        self.maskA = None
        self.maskB = None
        self.w, _, _ = im_size
        self.M = None
        self.points = None
        self.indices = None
        self.markers = {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up',
                        '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', '8': 'octagon',
                        's': 'square', 'p': 'pentagon', '*': 'star', '+': 'plus',
                        'x': 'x', 'd': 'thin_diamond'}
        self.markercolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.bboxMin = None
        self.bboxMax = None

    def sample_color(self, type="int"):
        if type == "int":
            return tuple(map(int, list(np.random.randint(0, 255, size=3))))
        else:
            return tuple(map(float, list(np.random.rand(3))))

    def get_transformation(self, points):
        m1 = np.mean(points, axis=0)

        angle = np.radians(np.random.randint(-25, 25, dtype=np.int16))
        sample_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)], [0., 0.]])
        signs = np.random.rand(2)
        signs[signs>0.5] = 1
        signs[signs != 1] = -1
        t = np.array(signs*( 10*np.random.randn(2) + 25), dtype=np.int32)
        translation = np.array([t[0] + m1[0] - m1[0] * np.cos(angle) + m1[1] * np.sin(angle),
                                t[1] + m1[1] - m1[0] * np.sin(angle) - m1[1] * np.cos(angle),
                                1.])
        M = np.c_[sample_rotation, translation]

        return M

    def sample_texture(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))

        ax.set_facecolor(self.sample_color(type="float"))
        fig.tight_layout(pad=0)
        w, h = np.subtract(fig.canvas.get_width_height()[::-1] ,(im_size[0], im_size[1]))
        area = (self.bboxMax[0] - self.bboxMin[0]) * (self.bboxMax[1] - self.bboxMin[1])
        size = 3 * int(np.log(area))
        X = np.random.randint(self.bboxMin[1]+w//2, self.bboxMax[1]+2*w, size=size)
        Y = np.random.randint(self.bboxMin[0]+h//2, self.bboxMax[0]+2*h, size=size)
        ax.margins(0)
        plt.axis('off')
        plt.xlim([0,350])
        plt.ylim([0,350])
        ax.add_artist(ax.patch)
        ax.patch.set_zorder(-1)
        ax.invert_yaxis()
        markerindex = np.random.randint(0, len(self.markers), size)
        for x,y in enumerate(self.markers):
            i = (markerindex == x)
            length, = X[i].shape
            colors = np.random.rand(length,3)
            t = matplotlib.markers.MarkerStyle(y)
            angle = np.random.randint(0,360)
            t._transform = t.get_transform().rotate_deg(angle)
            plt.scatter(X[i], Y[i], marker=t, c=colors, s=75*np.random.rand())

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        texture = self.crop_center(image_from_plot, im_size[0], im_size[1])
        fx, fy = np.gradient(np.pad(texture, ((0,1),(0,1), (0,0))), axis=[0, 1])
        fxx = np.gradient(fx, axis=0)
        fyy = np.gradient(fy, axis=1)
        laplacian = np.sum(fxx + fyy, axis=2)
        sign = np.sign(laplacian)
        diff_x = sign[:-1, :-1] - sign[:-1, 1:] < 0
        diff_y = sign[:-1, :-1] - sign[1:, :-1] < 0
        mask = np.logical_or(diff_x, diff_y)
        plt.close()
        return texture, mask

    def crop_center(self, img, cropx, cropy):
        y, x, c = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx]


class Circle(Shape):

    def __init__(self):
        super().__init__()

    def sample_radius(self):
        return np.random.randint(1, self.w//5)

    def sample_center(self):
        return np.random.randint(pad, self.w-pad, (2,), dtype=np.int32)

    def add_to_image(self, image, image2, flow, edgeField):
        center = self.sample_center()
        r = self.sample_radius()
        while np.any(center-r <0) or np.any(r+center>self.w):
            r = self.sample_radius()
        self.bboxMin = center-r
        self.bboxMax = center+r
        indices = np.indices(image[:, :, 0].shape).transpose(1, 2, 0)
        norm = np.linalg.norm(indices - center, axis=2)
        self.maskA = np.full(indices[:, :, 0].shape, False)
        self.maskA[np.where(norm < r)] = True
        ones = np.ones_like(indices[:, :, [0]])
        indices_image_hom = np.append(indices, ones, axis=2)
        self.M = self.get_transformation([(center[0], center[1])])

        tran_indices_hom = np.einsum('ij,klj->kli', self.M, indices_image_hom)
        tran_indices_normal = tran_indices_hom[:, :, :2] / tran_indices_hom[:, :, [-1]]
        indices_normal = indices.reshape((self.w, self.w, 2))

        flow_field = indices_normal - tran_indices_normal
        flow_field = flow_field.astype(np.int32)
        textureA, edges = self.sample_texture()
        warped = np.clip(tran_indices_normal.astype(np.int32), 0,im_size[0]-1).astype(np.uint16)
        war = (warped[:, :, 0], warped[:, :, 1])
        textureB = textureA[war]
        self.maskB = self.maskA[war]
        image[self.maskA] = textureA[self.maskA]
        image2[self.maskB] = textureB[self.maskB]
        flow[self.maskA] = flow_field[self.maskA]
        edgeField[self.maskA] = edges[self.maskA]
        return image, image2, flow, edgeField


class Polygon(Shape):

    def __init__(self):
        super().__init__()

    def sample_points(self):
        num_points = np.random.randint(3, 5)
        points = np.random.randint(pad, self.w-pad, (num_points, 2), dtype=np.int32)
        self.bboxMin = (np.min(points[:,0]),np.min(points[:,1]))
        self.bboxMax = (np.max(points[:,0]),np.max(points[:,1]))

        return points

    def add_to_image(self, image, image2, flow, edgeField):
        points = list(map(tuple, self.sample_points()))
        indices = np.indices(image[:, :, 0].shape).transpose(1, 2, 0).reshape((self.w * self.w, 2))
        p = Path(points)
        mask = p.contains_points(indices)
        self.points = indices[mask]
        self.maskA = mask.reshape((self.w, self.w))

        ones = np.ones(indices.shape[0])[:, None]
        indices_image_hom = np.append(indices, ones, axis=1).reshape((self.w, self.w, 3))
        self.M = self.get_transformation(points)

        tran_indices_hom = np.einsum('ij,klj->kli', self.M, indices_image_hom)
        tran_indices_normal = tran_indices_hom[:, :, :2] / tran_indices_hom[:, :, [-1]]
        indices_normal = indices.reshape((self.w, self.w, 2))

        flow_field = indices_normal - tran_indices_normal
        flow_field = np.round(flow_field).astype(np.int32)
        textureA, edges = self.sample_texture()
        warped = np.clip(tran_indices_normal.astype(np.int32), 0,im_size[0]-1).astype(np.uint16)
        war = (warped[:, :, 0], warped[:, :, 1])
        textureB = textureA[war]
        self.maskB = self.maskA[war]
        image[self.maskA] = textureA[self.maskA]
        image2[self.maskB] = textureB[self.maskB]
        flow[self.maskA] = flow_field[self.maskA]
        edgeField[self.maskA] = edges[self.maskA]

        return image, image2, flow, edgeField


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


path = "./Data"
im_size = (296,296, 3)
flow_size = (296, 296, 2)
pad = 20
blank = Image()

for i in tqdm(range(100)):
    drawImage(os.path.join(path, f'{str(i).zfill(3)}_img1.ppm'),
              os.path.join(path, f'{str(i).zfill(3)}_img2.ppm'),
              os.path.join(path, f'{str(i).zfill(3)}_flow.flo'),
              os.path.join(path, f'{str(i).zfill(3)}_flow.ppm'),
              os.path.join(path, f'{str(i).zfill(3)}_flow.ppm'))
