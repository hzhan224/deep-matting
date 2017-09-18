import torch.utils.data as data
from torchvision.transforms import ToTensor
from PIL import Image
from random import randrange


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# def load_img(filepath):
#     img = Image.open(filepath).convert('YCbCr')
#     y, _, _ = img.split()
#     return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, filelist):
        super(DatasetFromFolder, self).__init__()
        self.transform = ToTensor()
        with open(filelist) as f:
            self.image_filenames = f.readlines()

    def __getitem__(self, index):
        input = Image.open('data/train/input/' + self.image_filenames[index][:-1])
        trimap = Image.open('data/train/trimap/' + self.image_filenames[index][:-1])
        gt = Image.open('data/train/gt/' + self.image_filenames[index][:-1]).split()[0]

        w, h = input.size
        s = 320
        x, y = randrange(0, w-320), randrange(0, h-320)

        input = input.crop((x, y, x+320, y+320))
        trimap = trimap.crop((x, y, x+320, y+320))
        gt = gt.crop((x, y, x+320, y+320))

        input = self.transform(input)
        trimap = self.transform(trimap)
        gt = self.transform(gt)

        return input, trimap, gt

    def __len__(self):
        return len(self.image_filenames)
