import os
import sys
import datetime
import time
import numpy as np
import pdb
import sklearn.metrics as metrics

import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from dataloader.Mvtec_Loader import Mvtec_Dataloader
from networks.Controllable_Unet import Controllable_UNet
from networks.discriminator import Discriminator
from utils.vgg_loss import AdversarialLoss
from utils.visualizer import Visualizer
from utils.trick import adjust_lr, cuda_visible, print_args, save_ckpt, AverageMeter, LastAvgMeter
from utils.Canny import canny
from utils.parser import ParserArgs


class Stru_Model(nn.Module):
    def __init__(self, args):
        super(Stru_Model, self).__init__()
        self.args = args
        stru_net_skip_conn = [int(args.stru_net_skip_conn[i]) for i in range(len(args.stru_net_skip_conn))]
        model_Stru = Controllable_UNet(3, 2, stru_net_skip_conn, unit_channel=args.stru_unit_channel)
        model_D = Discriminator(in_channels=1)

        model_Stru = nn.DataParallel(model_Stru).cuda()
        model_D = nn.DataParallel(model_D).cuda()

        l1_loss = nn.L1Loss().cuda()
        adversarial_loss = AdversarialLoss().cuda()
        cross_entropy_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        self.add_module('model_Stru', model_Stru)
        self.add_module('model_D', model_D)

        self.add_module('mse_loss', mse_loss)
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('cross_entropy', cross_entropy_loss)

        # optimizer
        self.optimizer_Stru = torch.optim.Adam(params=self.model_Stru.parameters(),
                                               lr=args.lr,
                                               weight_decay=args.weight_decay,
                                               betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(params=self.model_D.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))

        if self.args.resume:
            ckpt_root = os.path.join(self.args.output_root, self.args.version, 'checkpoints')
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading Stru checkpoint '{}'".format(ckpt_path))
                checkpoint = torch.load(ckpt_path)

                args.start_epoch = checkpoint['epoch']
                self.model_Stru.load_state_dict(checkpoint['state_dict_Stru'])
                self.model_D.load_state_dict(checkpoint['state_dict_D'])

                print("=> loaded Stru and Discriminator checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(ckpt_path))

    def process(self, image):
        # process_outputs
        edge_map, edge_gt = self(image)
        edge_gt = edge_gt.cuda()
        edge = torch.max(edge_map, dim=1)[1].unsqueeze(1).float()
        edge_gt = edge_gt.max(dim=1)[0].unsqueeze(1).float()
        """
        Stru and D process, this package is reusable
        """
        # zero optimizers
        self.optimizer_Stru.zero_grad()
        self.optimizer_D.zero_grad()

        gen_loss = 0
        dis_loss = 0

        real = edge_gt
        fake = edge

        # discriminator loss
        dis_input_real = real
        dis_input_fake = fake.detach()
        dis_real, dis_real_feat = self.model_D(dis_input_real)
        dis_fake, dis_fake_feat = self.model_D(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += 0.001 * (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = fake
        gen_fake, gen_fake_feat = self.model_D(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.args.lamd_gen
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.args.lamd_fm
        gen_loss += gen_fm_loss

        gen_cross_entropy = self.cross_entropy(edge_map, edge_gt.squeeze(1).long()) * self.args.lamd_p
        # gen_cross_entropy = self.mse_loss(edge, edge_gt) * self.args.lamd_p
        gen_loss += gen_cross_entropy
        # create logs
        logs = dict(
            gen_gan_loss=gen_gan_loss,
            gen_fm_loss=gen_fm_loss,
            gen_cross_entropy_loss=gen_cross_entropy,
        )
        return edge, edge_gt, gen_loss, dis_loss, logs

    def forward(self, image):
        edge_gt = canny(image, self.args.canny_sigma).cuda()
        edge_map = self.model_Stru(image)

        return edge_map, edge_gt

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.optimizer_D.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.optimizer_Stru.step()


class RunMyModel(object):
    def __init__(self):
        args = ParserArgs().get_args()
        cuda_visible(args.gpu)

        cudnn.benchmark = True

        self.vis = Visualizer(env='{}'.format(args.version), port=args.port, server=args.vis_server)

        self.normal_train_loader, self.normal_test_loader, self.abnormal_loader =\
            Mvtec_Dataloader(data_root=args.mvtec_root,
                             batch=args.batch,
                             scale=args.scale,
                             category=args.data_modality,
                             crop_size=args.crop_size,
                             crop_rate=args.crop_rate).data_load()

        print_args(args)
        self.args = args
        self.new_lr = self.args.lr
        self.model = Stru_Model(args)
        self.train_diff_mean = None

        self.epoch = 0
        self.best_iou = 0
        self.is_best = False
        self.iou_train_last10 = LastAvgMeter(length=10)
        self.iou_val_abnormal_last10 = LastAvgMeter(length=10)
        self.iou_val_normal_last10 = LastAvgMeter(length=10)

        if args.predict:
            self.test()
        else:
            self.train_val()

    def train_val(self):
        self.vis.text(str(vars(self.args)), name='args')
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            adjust_lr_epoch_list = [400]
            adjust_lr(self.args.lr, self.model.optimizer_Stru, epoch, adjust_lr_epoch_list)
            adjust_lr(self.args.lr * self.args.d2g_lr, self.model.optimizer_D, epoch, adjust_lr_epoch_list)

            self.epoch = epoch
            self.train(epoch)

            if epoch % self.args.val_freq == 0 and epoch >= self.args.val_start_epoch:
                print('\nValidating')
                self.validate()

            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('Node: {}'.format(self.args.node))
            print('GPU: {}'.format(self.args.gpu))
            print('Version: {}\n'.format(self.args.version))

    def train(self, epoch):
        self.model.train()
        prev_time = time.time()
        train_loader = self.normal_train_loader
        for i, (image, _, _) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            # train
            edge, edge_gt, gen_loss, dis_loss, logs = self.model.process(image)
            # backward
            self.model.backward(gen_loss, dis_loss)

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = self.epoch * train_loader.__len__() + i
            batches_left = self.args.n_epochs * train_loader.__len__() - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
                             (self.epoch, self.args.n_epochs,
                              i, train_loader.__len__(),
                              dis_loss.item(),
                              gen_loss.item(),
                              time_left))

            # --------------
            #  Visdom
            # --------------
            if i == 0:
                image = image[:self.args.vis_batch]
                edge = edge[:self.args.vis_batch]
                edge_gt = edge_gt[:self.args.vis_batch]

                edge_vis = torch.cat([edge, edge, edge], dim=1)
                edge_gt_vis = torch.cat([edge_gt, edge_gt, edge_gt], dim=1)
                vim_images = torch.cat([image.cpu(), edge_vis.cpu(), edge_gt_vis.cpu()], dim=0)
                self.vis.images(vim_images, win_name='train', nrow=self.args.vis_batch)

            if i+1 == train_loader.__len__():
                self.vis.plot_multi_win(dict(dis_loss=dis_loss.item()))
                self.vis.plot_single_win(dict(gen_loss=gen_loss.item(),
                                              gen_cross_entropy_loss=logs['gen_cross_entropy_loss'].item(),
                                              gen_fm_loss=logs['gen_fm_loss'].item(),
                                              gen_gan_loss=logs['gen_gan_loss'].item()),
                                         win='gen_loss')

    def validate(self):
        # self.model.eval()
        self.model.eval()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            train_iou_list = self.forward_stru_dataloader(loader=self.normal_train_loader)
            abnormal_iou_list = self.forward_stru_dataloader(loader=self.abnormal_loader, mode='val_abnormal')
            normal_iou_list = self.forward_stru_dataloader(loader=self.normal_test_loader, mode='val_normal')

            # compute iou
            train_iou = torch.mean(train_iou_list)
            val_abnormal_iou = torch.mean(abnormal_iou_list)
            val_normal_iou = torch.mean(normal_iou_list)

            # update
            self.iou_train_last10.update(train_iou)
            self.iou_val_normal_last10.update(val_normal_iou)
            self.iou_val_abnormal_last10.update(val_abnormal_iou)

            self.is_best = val_normal_iou > self.best_iou
            self.best_iou = max(val_normal_iou, self.best_iou)

            """
            plot metrics curve
            """
            # total auc, primary metrics
            self.vis.plot_single_win(dict(value=train_iou,
                                          last_avg=self.iou_train_last10.avg,
                                          last_std=self.iou_train_last10.std),
                                     win='train iou')
            self.vis.plot_single_win(dict(value=val_normal_iou,
                                          best=self.best_iou,
                                          last_avg=self.iou_val_normal_last10.avg,
                                          last_std=self.iou_val_normal_last10.std),
                                     win='val normal iou')
            self.vis.plot_single_win(dict(value=val_abnormal_iou,
                                          last_avg=self.iou_val_abnormal_last10.avg,
                                          last_std=self.iou_val_abnormal_last10.std),
                                     win='val abnormal iou')

            metrics_str = 'iou_train_last20_avg = {:.4f}, iou_train_last20_std = {:.4f}, '\
                .format(self.iou_train_last10.avg, self.iou_train_last10.std)

            self.vis.text(metrics_str)

        save_ckpt(version=self.args.version,
                  state={
                      'epoch': self.epoch,
                      'state_dict_Stru': self.model.model_Stru.state_dict(),
                      'state_dict_D': self.model.model_D.state_dict(),
                  },
                  epoch=self.epoch,
                  is_best=self.is_best,
                  args=self.args)

        print('\n Save ckpt successfully!')
        print('\n', metrics_str)

    def test(self):
        self.model.train()
        iou_dict = {}
        threshold_list = np.linspace(0, 0.4, 100)
        for threshold in threshold_list:
            iou_dict[threshold] = AverageMeter()
        for i, (image, image_name_item, mask) in enumerate(self.abnormal_loader):
            image = image.cuda(non_blocking=True)
            # val, forward
            edge, image_rec = self.model(image)
            """
            preditction
            """
            image_residual = torch.abs(image_rec - image).max(dim=1)[0]
            mask = mask.cuda()
            for im in range(image.size(0)):
                for threshold in threshold_list:
                    region_mask = (image_residual[im] >= threshold).float()
                    iou = torch.sum(region_mask * mask[im, 0]) / torch.sum(((region_mask + mask[im, 0]) > 0).float())
                    iou_dict[threshold].update(iou)
        best_iou = 0
        best_threshold = 0
        iou_list = []
        for (key, item) in iou_dict.items():
            iou_list.append(item.avg)
            if item.avg > best_iou:
                best_iou = item.avg
                best_threshold = key
        self.vis.line(iou_list, threshold_list, win_name='iou_threshold_relationship')
        self.vis.text('best iou:{}, best_threshold:{}'.format(best_iou, best_threshold), name='best_iou')

    def forward_stru_dataloader(self, loader, mode='train'):
        iou_list = []

        for i, (image, image_name_item, mask) in enumerate(loader):
            image = image.cuda(non_blocking=True)
            # val, forward
            edge_map, edge_gt = self.model(image)
            edge = torch.max(edge_map, dim=1)[1].unsqueeze(1).float()
            edge_gt = torch.max(edge_gt, dim=1)[0].unsqueeze(1).float()
            iou_list += [torch.sum(edge[j] * edge_gt[j, 0]) /
                         torch.sum(((edge[j] + edge_gt[j, 0]) > 0).float()) for j in range(image.size(0))]

            if self.epoch % self.args.save_image_freq == 0:
                """
                save images
                """
                edge_save = torch.cat([edge] * 3, dim=1).cuda()
                edge_gt_save = torch.cat([edge_gt] * 3, dim=1).cuda()
                vim_images = torch.cat([image,
                                        edge_save,
                                        edge_gt_save],
                                       dim=0)

                output_save = os.path.join(self.args.output_root,
                                           '{}'.format(self.args.version),
                                           'Stru_sample')

                if not os.path.exists(output_save):
                    os.makedirs(output_save)
                tv.utils.save_image(vim_images, os.path.join(
                    output_save, '{}_{}_{}.png'.format(mode, self.epoch, i)), nrow=image.size(0))

            """
            visdom
            """
            if i == 0 and mode != 'train':
                image = image[:self.args.vis_batch]
                edge = edge[:self.args.vis_batch]
                edge_gt = edge_gt[:self.args.vis_batch]

                edge_vis = torch.cat([edge] * 3, dim=1).cuda()
                edge_gt_vis = torch.cat([edge_gt] * 3, dim=1).cuda()
                vim_images = torch.cat([image,
                                        edge_vis,
                                        edge_gt_vis],
                                       dim=0)
                self.vis.images(vim_images, win_name='{}'.format(mode), nrow=self.args.vis_batch)

        return torch.FloatTensor(iou_list)


if __name__ == '__main__':
    RunMyModel()
