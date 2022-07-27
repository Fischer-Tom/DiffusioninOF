import os
import glob
import random
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import imageio
from PIL import Image as PILImage, ImageDraw
import flow_vis
from scipy.signal import convolve2d
import cv2
import oflibnumpy as of


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def drawImage():
    image = Image(size)
    image.populate()
    feature_masked = np.zeros_like(image.flow)
    feature_masked[image.flow_features] = image.flow[image.flow_features]

    return image, feature_masked

class Image():

    def __init__(self, size):
        im_size = (size[0], size[1], 3)
        flow_size = (size[0], size[1], 2)
        self.dims = size
        self.image1 = np.zeros(im_size)
        self.image2 = np.zeros(im_size)
        self.flow = np.zeros(flow_size, dtype=np.float32)
        self.flow_features = np.full(size, False)

    def populate(self):
        shapes = [Polygon(self.dims),Circle(self.dims)]
        elements = np.random.randint(2, 6) * 4
        for _ in range(elements):
            shape = random.choice(shapes)
            shape.generateMasks()
            self.image1, self.image2, self.flow, self.flow_features = shape.add_to_image(self.image1, self.image2, self.flow, self.flow_features)


class Shape():

    def __init__(self, dims):
        self.M = None
        self.mask1 = None
        self.mask2 = None
        self.indices = np.indices(dims).transpose(1, 2, 0)
        self.indicesT = None
        self.w, self.h = dims
        self.features = Feature()
        self.flow_field = None

    def sample_color(self, type="int"):
        if type == "int":
            return tuple(map(int, list(np.random.randint(0, 255, size=3))))
        else:
            return tuple(map(float, list(np.random.rand(3))))

    def sample_flow(self, points):
        m1 = np.mean(points, axis=0, dtype=int)

        angle = np.random.randint(-45, 45, dtype=int)
        signs = np.random.rand(2)
        signs[signs > 0.5] = 1
        signs[signs != 1] = -1
        t = np.array(signs * (10 * np.random.randn(2) + 25), dtype=int)

        flow = of.Flow.from_transforms([['rotation', int(m1[0]), int(m1[1]), angle],['translation',int(t[0]),int(t[1])]], (self.w, self.h), 't')
        return flow

    def add_to_image(self, image1, image2, flow, flow_features):

        color = self.sample_color()

        empty_image = np.zeros_like(image1)
        empty_image[self.mask1] = color
        warped_image = self.flow_field.apply(empty_image).astype(np.uint8)
        mask2 = warped_image == color
        filled_image, flow_features = self.add_features(empty_image, flow_features)

        warped_image = self.flow_field.apply(filled_image)

        image1[self.mask1] = filled_image[self.mask1]
        image2[mask2] = warped_image[mask2]

        flow[self.mask1] = self.flow_field.vecs[self.mask1]

        if np.count_nonzero(self.mask1) < (35*35):
            flow_features[self.mask1] = True
        return image1, image2, flow, flow_features

    def add_features(self, image1, flow_features):
        w_i, h_i, _ = image1.shape
        amount = np.random.randint(5,8)
        flow_features[self.mask1] = False
        valid_indices = self.indices[self.mask1]
        if valid_indices.shape[0] == 0:
            return image1, flow_features
        for _ in range(amount):
            feature_mask = self.features.get_feature_mask()
            w, h = feature_mask.shape
            color = self.sample_color()
            o_x, o_y = random.choice(valid_indices)
            odd = w % 2
            if odd:
                if np.all(self.mask1[o_x - w // 2 - 1:o_x + w // 2, o_y - h // 2 - 1:o_y + h // 2]):
                    ind_mask = self.indices[o_x - w // 2 - 1:o_x + w // 2, o_y - h // 2 - 1:o_y + h // 2]
                    if ind_mask.shape[:2] == feature_mask.shape:
                        ind = ind_mask[feature_mask]
                        ind = (ind[:, 0], ind[:, 1])
                        image1[ind] = color
                        flow_features[ind] = True

            else:
                if np.all(self.mask1[o_x - w // 2:o_x + w // 2, o_y - h // 2:o_y + h // 2]):
                    ind_mask = self.indices[o_x - w // 2:o_x + w // 2, o_y - h // 2:o_y + h // 2]
                    if ind_mask.shape[:2] == feature_mask.shape:
                        ind = ind_mask[feature_mask]
                        ind = (ind[:, 0], ind[:, 1])
                        image1[ind] = color
                        flow_features[ind] = True

        return image1, flow_features


class Circle(Shape):
    def __init__(self, dims):
        super().__init__(dims)

    def sample_radius(self):
        return np.random.randint(1, self.w // (5*4))

    def sample_center(self):
        return np.random.randint(0, self.w, (2,), dtype=np.int32)

    def generateMasks(self):
        center = self.sample_center()
        r = self.sample_radius()
        while np.any(center - r < 0) or np.any(r + center > self.w):
            center = self.sample_center()
            r = self.sample_radius()
        norm = np.linalg.norm(self.indices - center, axis=2)
        self.mask1 = np.full((self.w, self.h), False)
        self.mask1[np.where(norm < r)] = True
        self.flow_field = self.sample_flow([(center[0], center[1])])



class Polygon(Shape):
    def __init__(self, dims):
        super().__init__(dims)

    def sample_points(self):
        num_points = np.random.randint(3, 5)
        area = np.random.randint(0, self.w, 2, dtype=np.int32)
        max_diff = self.w // 6
        pointsX = np.random.randint(np.maximum(0,area[0]-max_diff), np.minimum(area[0]+max_diff,self.w), (num_points), dtype=np.int16)
        pointsY = np.random.randint(np.maximum(0,area[1]-max_diff), np.minimum(area[1]+max_diff,self.h), (num_points), dtype=np.int16)
        points = np.stack((pointsX,pointsY), axis=1)

        return points

    def generateMasks(self):
        points = self.sample_points()
        img = PILImage.new('L', (self.w, self.h), 0)
        ImageDraw.Draw(img).polygon(list(points.flatten()), outline=1, fill=1)
        self.mask1 = np.array(img, dtype=bool)
        img.close()
        self.flow_field = self.sample_flow(points)

class Feature:

    def __init__(self):
        self.masks = []
        self.init_masks()

    def init_masks(self):
        for filename in glob.glob('Features/*.png'):
            im = PILImage.open(filename)
            mask = np.alltrue(np.array(im) != [0, 0, 0], axis=2)
            w,h = mask.shape
            border = 0
            temp_mask = mask.copy()
            for i in range(w):
                temp_mask[i:w-i,i:h-i] = False
                if np.alltrue(temp_mask==False):
                    border = i
                    temp_mask = mask.copy()
                    continue
                mask = mask[border:w-border,border:h-border]
                break
            self.masks.append(mask)

    def get_feature_mask(self):
        shape = random.choice(self.masks)
        rotation = np.random.randint(0, 359)
        shape_r = scipy.ndimage.rotate(shape, rotation, reshape=True,output=np.int16)

        size = np.random.randint(20, 30)
        downsampled_shape = cv2.resize(shape_r,dsize=(size, size),interpolation = cv2.INTER_AREA ) > 0

        return downsampled_shape


path = "./Data"
size = (1024, 1024)

for i in tqdm(range(0, 20, 4)):
    image, feature_masked = drawImage()
    x_dim = [(0, size[0]//2),(size[0]//2, size[0]), (0, size[0]//2), (size[0]//2, size[0])]
    y_dim = [(0, size[1]//2),(0, size[1]//2),(size[1]//2, size[1]),(size[1]//2, size[1])]
    for j, ((a,b),(c,d)) in enumerate(zip(x_dim, y_dim)):
        name1 = os.path.join(path, f'{str(i+j).zfill(3)}_img1.ppm')
        name2 = os.path.join(path, f'{str(i+j).zfill(3)}_img2.ppm')
        flowname = os.path.join(path, f'{str(i+j).zfill(3)}_flow.flo')
        edgeName = os.path.join(path, f'{str(i+j).zfill(3)}_features_flow.ppm')
        flowColorname = os.path.join(path, f'{str(i+j).zfill(3)}_flow.ppm')

        imageio.imwrite(name1, image.image1[a:b,c:d,:].astype(np.uint8))
        imageio.imwrite(name2, image.image2[a:b,c:d,:].astype(np.uint8))
        imageio.imwrite(edgeName, flow_vis.flow_to_color(feature_masked[a:b,c:d,:]).astype(np.uint8))
        imageio.imwrite(flowColorname, flow_vis.flow_to_color(image.flow[a:b,c:d,:]))
        writeFlow(flowname, image.flow[a:b,c:d,:])
