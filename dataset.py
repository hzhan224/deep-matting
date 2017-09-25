import torch.utils.data as data
from torchvision.transforms import ToTensor
from PIL import Image, ImageFilter
from random import randrange
from glob import glob
import numpy as np


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# def load_img(filepath):
#     img = Image.open(filepath).convert('YCbCr')
#     y, _, _ = img.split()
#     return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, filelist):
        super(DatasetFromFolder, self).__init__() #fuleijicheng
        self.transform = ToTensor()
        with open(filelist) as f:
            self.image_filenames = f.readlines()
        self.bg_list = glob('data/train/bg/*.jpg')
        self.factor = len(self.bg_list)
        

    def __getitem__(self, index):
        input = Image.open('data/train/input/' + self.image_filenames[index][:-1])
        trimap = Image.open('data/train/trimap/' + self.image_filenames[index][:-1])
        gt = Image.open('data/train/gt/' + self.image_filenames[index][:-1])
        bg = Image.open(choice(self.bg_list))

        w, h = input.size
        s = 320
        x, y = randrange(0, w-320), randrange(0, h-320)

        input = input.crop((x, y, x+320, y+320))
        trimap = trimap.crop((x, y, x+320, y+320))
        gt = gt.crop((x, y, x+320, y+320))

        bx, by, rad = randrange(0, w-320), randrange(0, h-320), randrang(0, 10)
        bg = bg.crop((bx, by, bx+320, by+320)).filter(ImageFilter.GaussianBlur(rad))

        fg = np.array(input) * (np.array(gt) / 255.0)

        input = fg + np.array(bg) * (1 - np.array(gt) / 255.0)
        input = Image.fromarray(input)
        fg = Image.fromarray(fg)
        
        input = self.transform(input)
        trimap = self.transform(trimap)
        fg = self.transform(fg)
        bg = self.transform(bg)
        gt = self.transform(gt)

        return input, trimap, gt, fg, bg

    def __len__(self):
        return len(self.image_filenames) * self.factor
