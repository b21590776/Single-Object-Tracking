import numpy as np
import math
from skimage import transform, img_as_ubyte
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")


class Rescale(object):
    # Rescaling image and bounding box.
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
           if h > w:
               new_h, new_w = self.output_size*h/w, self.output_size
           else:
               new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        img = img_as_ubyte(img)
        bb = [bb[0]*new_w/w, bb[1]*new_h/h, bb[2]*new_w/w, bb[3]*new_h/h]
        return {'image': img, 'bb':bb}


class CropPrevious(object):
    # Cropping the previous frame image using the bounding box specifications.
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bb = sample['image'], sample['bb']
        image = img_as_ubyte(image)
        if (len(image.shape) == 2):
            image = np.repeat(image[...,None],3,axis=2)
        im = Image.fromarray(image)
        w = bb[2]-bb[0]
        h = bb[3]-bb[1]
        left = bb[0]-w/2
        top = bb[1]-h/2
        right = left + 2*w
        bottom = top + 2*h
        box = (left, top, right, bottom)
        box = tuple([int(math.floor(x)) for x in box])
        res = np.array(im.crop(box))
        bb = [bb[0]-left, bb[1]-top, bb[2]-left, bb[3]-top]
        return {'image':res, 'bb':bb}


class CropCurrent(object):
    # Crop the current frame image using the bounding box specifications.
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, prevbb, currbb = sample['image'], sample['prevbb'], sample['currbb']
        image = img_as_ubyte(image)
        if (len(image.shape) == 2):
            image = np.repeat(image[...,None],3,axis=2)
        im = Image.fromarray(image)
        w = prevbb[2]-prevbb[0]
        h = prevbb[3]-prevbb[1]
        left = prevbb[0]-w/2
        top = prevbb[1]-h/2
        right = left + 2*w
        bottom = top + 2*h
        box = (left, top, right, bottom)
        box = tuple([int(math.floor(x)) for x in box])
        res = np.array(im.crop(box))
        bb = [currbb[0]-left, currbb[1]-top, currbb[2]-left, currbb[3]-top]
        return {'image':res, 'bb':bb}

class ToTensor(object):
    # Converting ndarrays in sample to Tensors.
    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        prev_img = prev_img.transpose((2, 0, 1))
        curr_img = curr_img.transpose((2, 0, 1))
        if 'currbb' in sample:
            currbb = sample['currbb']
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float(),
                    'currbb': torch.from_numpy(currbb).float()
                    }
        else:
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float()
                    }


class Normalize(object):
    # returning image with zero mean and scales bounding box by factor of 10.
    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.mean = [112, 112,112]
        prev_img = prev_img.astype(float)
        curr_img = curr_img.astype(float)
        prev_img -= np.array(self.mean).astype(float)
        curr_img -= np.array(self.mean).astype(float)
        if 'currbb' in sample:
            currbb = sample['currbb']
            currbb = currbb * (10. / 224)
            return {'previmg': prev_img,
                    'currimg': curr_img,
                    'currbb': currbb
                    }
        else:
            return {'previmg': prev_img,
                    'currimg': curr_img
                    }


# return the current bouding box values in the orignal image dimensions
def transformbb(currbb, prevbb):
    # unscaling
    w = (prevbb[2] - prevbb[0]) * 2
    h = (prevbb[3] - prevbb[1]) * 2
    # input image size to network
    net_w = 224
    net_h = 224
    bb2 = [currbb[0]*w/net_w,
                  currbb[1]*h/net_h,
                  currbb[2]*w/net_w,
                  currbb[3]*h/net_h]
    # uncropping
    bb = prevbb
    w = bb[2]-bb[0]
    h = bb[3]-bb[1]
    left = bb[0]-w/2
    top = bb[1]-h/2
    currbb = [left+bb2[0], top+bb2[1], left+bb2[2], top+bb2[3]]
    return currbb

