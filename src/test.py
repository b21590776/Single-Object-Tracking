import os
from src import model
from torch.autograd import Variable
from torchvision import transforms
from src.operations import *
from skimage import io, transform
import matplotlib.patches as patches
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()

trained_model = '../save/pa3/model_n_epoch_5.pth'  # path to trained model
data_directory = '../dataset/videos/test'          # path to video frames

class Test:
    def __init__(self, vid_src, model_path):
        self.root_dir = vid_src
        self.transform = transforms.Compose([Normalize(), ToTensor()])
        self.model_path = model_path
        self.model = model.ConNet()
        if use_gpu:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        frames = os.listdir(vid_src)
        frames = [vid_src + "/" + frame for frame in frames]
        self.len = len(frames)-1
        frames = np.array(frames)
        frames.sort()
        self.x = []
        for i in range(self.len):
            self.x.append([frames[i], frames[i+1]])
        self.x = np.array(self.x)
        f = open("../dataset/annotations/"+vid_src[-9:]+".ann")
        lines = f.readlines()
        init_bbox = lines[0].strip().split(' ')
        init_bbox = [float(x) for x in init_bbox]
        init_bbox = [init_bbox[1], init_bbox[2], init_bbox[3], init_bbox[4]]
        init_bbox = np.array(init_bbox)
        print("initial box: ", init_bbox)
        self.prev_rect = init_bbox

    # returning transformed pytorch tensor which is passed directly to the network
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return self.transform(sample)

    # returning cropped and scaled previous frame and current frame in numpy format
    def get_sample(self, idx):
        prev = io.imread(self.x[idx][0])
        curr = io.imread(self.x[idx][1])
        prevbb = self.prev_rect
        crop_prev = CropPrevious(224)
        scale = Rescale((224, 224))
        transform_prev = transforms.Compose([crop_prev, scale])
        prev_img = transform_prev({'image': prev, 'bb':prevbb})['image']
        curr_img = transform_prev({'image':curr, 'bb':prevbb})['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        return sample

    # given previous frame and next frame, regressing the bounding box coordinates
    def get_rectangle(self, sample):
        x1, x2 = sample['previmg'], sample['currimg']
        if use_gpu:
            x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
        else:
            x1, x2 = Variable(x1), Variable(x2)
        x1 = x1[None, :, :, :]
        x2 = x2[None, :, :, :]
        y = self.model(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bb = list(bb * (224. / 10))
        bb = transformbb(bb, self.prev_rect)
        return bb

    # loop through all the frames of test sequence and tracking target object
    def test(self):
        fig, ax = plt.subplots(1)
        for i in range(self.len):
            sample = self[i]
            bb = self.get_rectangle(sample)
            im = io.imread(self.x[i][1])
            # show rectangle
            ax.clear()
            ax.imshow(im)
            rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor='g',
                                     facecolor='none')
            ax.add_patch(rect)
            self.prev_rect = bb
            print("frame: {}".format(i + 1), bb)
        plt.show()


if __name__ == "__main__":
    videos = os.listdir(data_directory)
    for video in videos:
        vid_src = data_directory + "/" + video
        tester = Test(vid_src, trained_model)
        tester.test()
