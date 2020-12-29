from PIL import Image
import glob
import os
import pdb
import torch
import torch.utils.data as data
import torchvision.transforms as T

# without data augmentation
# RGB to gray

category_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill',
                 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class Mvtec_Dataloader(object):
    def __init__(self, data_root, batch, scale, category, crop_size=None, crop_rate=None, flip_rate=0.5):
        if category not in category_list:
            raise ValueError('wrong category:{}'.format(category))

        self.data_root = os.path.join(data_root, category)

        image_crop_transform = T.Compose([
            T.RandomCrop((crop_size, crop_size)),
            T.RandomHorizontalFlip(flip_rate),
            T.Resize((scale, scale), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

        image_transform = T.Compose([
            T.Resize((scale, scale), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

        self.config = dict(
            image_crop_transform=image_crop_transform,
            image_transform=image_transform,
            batch=batch,
            crop_rate=crop_rate
        )

    def data_load(self):
        normal_train_loader = data.DataLoader(
            dataset=NormalTrain_Dataset(
                    data_root=self.data_root,
                    config=self.config),
            batch_size=self.config['batch'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        normal_test_loader = data.DataLoader(
            dataset=NormalTest_Dataset(
                data_root=self.data_root,
                config=self.config),
            batch_size=self.config['batch'],
            num_workers=8,
            pin_memory=True,
        )

        abnormal_loader = data.DataLoader(
            dataset=Abnormal_Dataset(
                data_root=self.data_root,
                config=self.config),
            batch_size=self.config['batch'],
            num_workers=8,
            pin_memory=True,
            shuffle=True
        )

        return normal_train_loader, normal_test_loader, abnormal_loader


class BaseFundusDataset(data.Dataset):
    def __init__(self, data_root, config, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.images_path_list = None
        self.masks_ok = None
        self.config = config

    def __getitem__(self, item):
        image_path = self.images_path_list[item]
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path).convert('RGB')

        seed = torch.rand(1).item()
        if seed >= self.config['crop_rate'] or (self.mode == 'val'):
            image = self.config['image_transform'](image)
        else:
            image = self.config['image_crop_transform'](image)

        if self.masks_ok:
            mask_path = image_path.replace('test', 'ground_truth').replace('.png', '_mask.png')
            mask = Image.open(mask_path).convert('L')
            mask = self.config['image_transform'](mask)
        else:
            mask = image

        return image, image_name, mask

    def __len__(self):
        return len(self.images_path_list)


class NormalTrain_Dataset(BaseFundusDataset):
    def __init__(self, data_root, config):
        # TODO: optimize
        super(NormalTrain_Dataset, self).__init__(data_root, config)
        self.images_path_list = glob.glob('{}/train/good/*'.format(data_root))
        self.mode = 'train'


class NormalTest_Dataset(BaseFundusDataset):
    def __init__(self, data_root, config):
        super(NormalTest_Dataset, self).__init__(data_root, config)
        self.images_path_list = glob.glob('{}/test/good/*'.format(data_root))
        self.mode = 'val'


class Abnormal_Dataset(BaseFundusDataset):
    def __init__(self, data_root, config):
        super(Abnormal_Dataset, self).__init__(data_root, config)
        self.mode = 'val'
        abnormal_category = glob.glob('{}/test/*'.format(data_root))
        abnormal_category = [os.path.split(abnormal_category[i])[-1] for i in range(len(abnormal_category))]
        abnormal_category.remove('good')
        self.images_path_list = []
        self.masks_path_list = []
        for i in range(len(abnormal_category)):
            self.images_path_list += glob.glob('{}/test/{}/*'.format(data_root, abnormal_category[i]))
        self.masks_ok = True


if __name__ == '__main__':
    import pdb

    data_root = '/home/imed/new_disk/imed_dataset/iSee_anomaly/preprocess'
    config = dict(transform=None)


