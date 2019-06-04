import os
from src.operations import *
from skimage import io, transform
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Datasets(Dataset):
    def __init__(self, root_dir, target_dir, transform=None):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.y = []
        self.x = []
        self.transform = transform
        videos = os.listdir(root_dir)
        for video in videos:
            env_videos = os.listdir(root_dir + video)
            vid_src = self.root_dir + video
            vid_ann = self.target_dir + video + ".ann"
            frames = [vid_src + "/" + vid for vid in env_videos]
            f = open(vid_ann, "r")
            annotations = f.readlines()
            f.close()
            frames = np.array(frames)
            for i in range(len(frames)-1):
                self.x.append([frames[i], frames[i+1]])
                self.y.append([annotations[i], annotations[i+1]])
        self.len = len(self.y)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    # returning size of dataset
    def __len__(self):
        return self.len

    # returning transformed sample
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        if(self.transform):
            sample = self.transform(sample)
        return sample

    # returning sample without transformation for visualization
    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.get_bbox(self.y[idx][0])
        currbb = self.get_bbox(self.y[idx][1])
        crop_prev = CropPrevious(224)
        crop_curr = CropCurrent(224)
        scale = Rescale((224, 224))
        transform_prev = transforms.Compose([crop_prev, scale])
        prev_img = transform_prev({'image': prev, 'bb': prevbb})['image']
        curr_obj = crop_curr({'image': curr, 'prevbb': prevbb, 'currbb': currbb})
        curr_obj = scale(curr_obj)
        curr_img = curr_obj['image']
        currbb = curr_obj['bb']
        currbb = np.array(currbb)
        sample = {'previmg': prev_img, 'currimg': curr_img,'currbb' : currbb}
        return sample

    # given annotation returning bounding box in the format: (left, upper, width, height)
    def get_bbox(self, ann):
        ann = ann.strip().split(' ')
        left = float(ann[1])
        top = float(ann[2])
        right = float(ann[3])
        bottom = float(ann[4])
        return [left, top, right, bottom]


