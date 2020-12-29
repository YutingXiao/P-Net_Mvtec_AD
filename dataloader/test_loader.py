# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       test_loader
   Project Name:    oct_fundus_ano
   Author :         Kang ZHOU
   Date:            2019/7/9
-------------------------------------------------
   Change Activity:
                   2019/7/9:
-------------------------------------------------
"""
import os

from PIL import Image
from scipy import misc

import torch.utils.data as data
import torchvision.transforms as T


class Test_Loader(object):
    def __init__(self, data_root, folder, batch, scale, workers=8, shuffle=False):
        self.data_root = data_root
        self.folder = folder
        self.batch = batch
        self.workers = workers
        self.config = dict(
            scale=scale,
            shuffle=shuffle
        )

    def data_load(self):
        test_set = Test_Dataset(
            data_root=self.data_root,
            test_folder=self.folder,
            config=self.config
        )

        drive_vessel_loader = data.DataLoader(
            dataset=test_set,
            batch_size=self.batch,
            num_workers=self.workers,
            pin_memory=True,
            shuffle=self.config['shuffle']
        )

        return drive_vessel_loader


class Test_Dataset(data.Dataset):
    def __init__(self, data_root, test_folder, config):
        super(Test_Dataset, self).__init__()

        self.data_root = os.path.join(data_root, test_folder)
        self.images_name_list = os.listdir(self.data_root)

        # images transform
        self.t = T.Compose([
            T.Resize((config['scale'], config['scale'])),
            T.ToTensor()
        ])

    def __getitem__(self, item):
        image_name_item = self.images_name_list[item]
        image_path = os.path.join(self.data_root, image_name_item)

        # green channel
        image = Image.fromarray(misc.imread(image_path)[:, :, 1])

        if self.t is not None:
            image = self.t(image)

        return image, image_name_item

    def __len__(self):
        return len(self.images_name_list)
