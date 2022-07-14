import os
import glob
import random
import numpy as np
import scipy.ndimage
from tqdm import tqdm
import imageio
from PIL import Image as PILImage, ImageDraw
import flow_vis

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

def drawImage(name1, name2, flowname, edgeName, flowColorname=None):

    image = Image(size)
    image.populate()

    imageio.imwrite(name1, image.image1.astype(np.uint8))
    imageio.imwrite(name2, image.image2.astype(np.uint8))
    #imageio.imwrite(edgeName, edgeField.astype(np.uint8))
    imageio.imwrite(flowColorname, flow_vis.flow_to_color(image.flow))
    writeFlow(flowname, image.flow)

class Image():

    def __init__(self, size):
        im_size = (size[0], size[1], 3)
        flow_size = (size[0], size[1], 2)
        self.dims = size
        self.image1 = np.zeros(im_size)
        self.image2 = np.zeros(im_size)
        self.flow = np.zeros(flow_size, dtype=np.float32)
        self.edgeField = np.zeros((flow_size[0], flow_size[1]), dtype=np.int8)

    def populate(self):
        shapes = [Polygon(self.dims),Polygon(self.dims),Polygon(self.dims),Circle(self.dims)]
        elements = np.random.randint(2,6)
        for _ in range(elements):
            shape = random.choice(shapes)
            shape.generateMasks()
            shape.add_to_image(self.image1, self.image2, self.flow)
            shape.add_features(self.image1, self.image2)

class Shape():

    def __init__(self, dims):
        self.M = None
        self.mask1 = None
        self.mask2 = None
        self.indices = np.indices(dims).transpose(1, 2, 0)
        self.indicesT = None
        self.w, self.h = dims
        self.features = Feature()

    def sample_color(self, type="int"):
        if type == "int":
            return tuple(map(int, list(np.random.randint(0, 255, size=3))))
        else:
            return tuple(map(float, list(np.random.rand(3))))

    def sample_transformation(self, points):
        m1 = np.mean(points, axis=0)

        angle = np.radians(np.random.randint(-25, 25, dtype=np.int16))
        sample_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)], [0., 0.]])
        signs = np.random.rand(2)
        signs[signs > 0.5] = 1
        signs[signs != 1] = -1
        t = np.array(signs*( 10*np.random.randn(2) + 25), dtype=np.int32)
        translation = np.array([t[0] + m1[0] - m1[0] * np.cos(angle) + m1[1] * np.sin(angle),
                                t[1] + m1[1] - m1[0] * np.sin(angle) - m1[1] * np.cos(angle),
                                1.])
        M = np.c_[sample_rotation, translation]

        return M

    def add_to_image(self, image1, image2, flow):

        color = self.sample_color()

        flow_field = self.indices - self.indicesT
        flow_field = flow_field.astype(np.int32)

        image1[self.mask1] = color
        image2[self.mask2] = color
        flow[self.mask1] = flow_field[self.mask1]

        return image1, image2, flow

    def add_features(self, image1, image2):
        feature_mask = self.features.get_feature_mask()
        color = self.sample_color()
        valid_indices = self.indices[self.mask1]
        position = random.choice(valid_indices)

        print("AV")

class Circle(Shape):
    def __init__(self, dims):
        super().__init__(dims)

    def sample_radius(self):
        return np.random.randint(1, self.w//5)

    def sample_center(self):
        return np.random.randint(0, self.w, (2,), dtype=np.int32)

    def generateMasks(self):
        center = self.sample_center()
        r = self.sample_radius()
        while np.any(center-r <0) or np.any(r+center>self.w):
            center = self.sample_center()
            r = self.sample_radius()
        norm = np.linalg.norm(self.indices - center, axis=2)
        self.mask1 = np.full((self.w, self.h), False)
        self.mask1[np.where(norm < r)] = True
        self.M = self.sample_transformation([(center[0], center[1])])
        ones = np.ones_like(self.indices[:, :, [0]])
        indices_image_hom = np.append(self.indices, ones, axis=2)
        tran_indices_hom = np.einsum('ij,klj->kli', self.M, indices_image_hom)
        self.indicesT = (tran_indices_hom[:, :, :2] / tran_indices_hom[:, :, [-1]]).astype(np.int32)
        self.indices = self.indices.reshape((self.w, self.w, 2))
        new_center = self.indicesT[center[0],center[1],:]
        norm = np.linalg.norm(self.indices - new_center, axis=2)
        self.mask2 = np.full((self.w, self.h), False)
        self.mask2[np.where(norm < r)] = True

class Polygon(Shape):
    def __init__(self, dims):
        super().__init__(dims)

    def sample_points(self):
        num_points = np.random.randint(3, 5)
        points = np.random.randint(0, self.w, (num_points, 2), dtype=np.int32)

        return points

    def generateMasks(self):

        points = self.sample_points()
        img = PILImage.new('L', (self.w,self.h),0)
        ImageDraw.Draw(img).polygon(list(points.flatten()), outline=1, fill=1)
        self.mask1 = np.array(img, dtype=bool)
        img.close()
        self.M = self.sample_transformation(points)
        ones = np.ones_like(self.indices[:, :, [0]])
        indices_image_hom = np.append(self.indices, ones, axis=2)
        tran_indices_hom = np.einsum('ij,klj->kli', self.M, indices_image_hom)
        self.indicesT = (tran_indices_hom[:, :, :2] / tran_indices_hom[:, :, [-1]]).astype(np.int32)
        self.indices = self.indices.reshape((self.w, self.w, 2))


        ones = np.ones_like(points[:, [0]])
        points_image_hom = np.append(points, ones, axis=1)
        tran_points_hom = np.einsum('ij,lj->li', self.M, points_image_hom)
        pointsT = (tran_points_hom[:, :2] / tran_points_hom[:, [-1]]).astype(np.int32)
        img = PILImage.new('L', (self.w, self.h), 0)
        ImageDraw.Draw(img).polygon(list(pointsT.flatten()), outline=1, fill=1)
        self.mask2 = np.array(img, dtype=bool)
        img.close()

class Feature:

    def __init__(self):
        self.masks = []
        self.init_masks()

    def init_masks(self):
        for filename in glob.glob('Shapes/*.png'):
            im = PILImage.open(filename)
            self.masks.append(np.alltrue(np.array(im) == [0,0,0], axis=2))

    def get_feature_mask(self):
        shape = random.choice(self.masks)
        rotation = np.random.randint(0,359)
        shape_r = scipy.ndimage.rotate(shape, rotation, reshape=True)

        return shape_r


path = "./Data"
size = (256,256)

for i in tqdm(range(1)):
    drawImage(os.path.join(path, f'{str(i).zfill(3)}_img1.ppm'),
              os.path.join(path, f'{str(i).zfill(3)}_img2.ppm'),
              os.path.join(path, f'{str(i).zfill(3)}_flow.flo'),
              os.path.join(path, f'{str(i).zfill(3)}_flow.ppm'),
              os.path.join(path, f'{str(i).zfill(3)}_flow.ppm'))


