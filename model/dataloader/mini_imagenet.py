import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from .simclr_view_generator import ContrastiveLearningViewGenerator

from model import file_dir
IMAGE_PATH1 = file_dir.mini_img
SPLIT_PATH = file_dir.mini_split
CACHE_PATH = file_dir.mini_cache

def get_simclr_pipeline_transform(size, s=1, extra_transforms=None):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    print('here ', extra_transforms)
    data_transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          *extra_transforms
    )
    return data_transforms

class MiniImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False, return_id=False, 
                return_simclr=None):
        im_size = args.im_size
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        self.csv = pd.read_csv(csv_path)
        self.setname = setname
        cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

        self.use_im_cache = args.use_im_cache
        self.return_id = return_id
        print('Inside Mini imagenet ', [self.use_im_cache, args.im_size])
        if self.use_im_cache:
            print('cache_apth ', cache_path)
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path)
                self.path = data
                self.data = [resize_(Image.open(path).convert('RGB')) for path in data ]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label, 'path':self.path}, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data  = cache['data']
                self.label = cache['label']
                self.path = cache['path']
        else:
            self.data, self.label = self.parse_csv(csv_path)

        self.num_class = len(set(self.label))

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(image_size),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        self.norm = transforms.Normalize(np.array([0.485, 0.456, 0.406]),np.array([0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transforms_list + [self.norm])
        self.return_simclr = return_simclr

        if self.return_simclr is not None:

            extra_transforms = [ transforms.Resize(image_size),
                  transforms.CenterCrop(image_size), self.norm]
            if self.use_im_cache: 
                self.init_transform = transforms.Compose([transforms.ToTensor()])
                self.simclr_transform = ContrastiveLearningViewGenerator(
                    get_simclr_pipeline_transform(im_size,
                    extra_transforms=extra_transforms), n_views=self.return_simclr)
            else:
                self.init_transform = transforms.Compose([transforms.Resize(128),
                    transforms.ToTensor()])
                self.simclr_transform = ContrastiveLearningViewGenerator(
                    get_simclr_pipeline_transform(128,
                    extra_transforms=extra_transforms), n_views=self.return_simclr)

    def parse_csv(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = data
        else:
            image = Image.open(data).convert('RGB')

        if self.return_simclr is not None:
            simclr_images = self.simclr_transform(self.init_transform(image))
            simclr_images = [s.unsqueeze(0) for s in simclr_images]
            simclr_images = torch.cat(simclr_images, dim=0)
        image = self.transform(image)

        if self.return_id:
            if self.use_im_cache:
                path = self.path[i]
                if self.return_simclr is not None:
                    return image, label, os.path.basename(path)[:-len('.jpg')], simclr_images
                else:
                    return image, label, os.path.basename(path)[:-len('.jpg')]
            else:
                if self.return_simclr is not None:
                    return image, label, os.path.basename(data)[:-len('.jpg')], simclr_images
                else:
                    return image, label, os.path.basename(data)[:-len('.jpg')]
        
        if self.return_simclr is not None:
            return image, label, simclr_images

        return image, label
