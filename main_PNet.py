import os
import sys
import datetime
import time
import numpy as np

import sklearn.metrics as metrics

import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from dataloader.Mvtec_Loader import Mvtec_Dataloader
from networks.MLNet_mask import MultiLevelNet
from networks.Controllable_Unet import Controllable_UNet
from networks.discriminator import Discriminator
from utils.vgg_loss import AdversarialLoss
from utils.visualizer import Visualizer
from utils.trick import *
from utils.Canny import canny
from utils.imge_blurr import image_blurr
from utils.dip_operation import dilated_eroded
from utils.parser import ParserArgs
from torchvision.models.resnet import resnet18


class PNetModel(nn.Module):
    def __init__(self, args, ablation_mode=4):
        super(PNetModel, self).__init__()
        self.args = args

        args.image_skip_conn = [int(args.image_skip_conn[i]) for i in range(len(args.image_skip_conn))]
        model_G = MultiLevelNet(in_ch=3, modality=self.args.data_modality, ablation_mode=ablation_mode,
                                 image_skip_conn=args.image_skip_conn)
        model_D = Discriminator(in_channels=3)

        model_G = nn.DataParallel(model_G).cuda()
        model_D = nn.DataParallel(model_D).cuda()

        l1_loss = nn.L1Loss().cuda()
        l2_loss = nn.MSELoss().cuda()
        adversarial_loss = AdversarialLoss().cuda()

        self.add_module('model_G', model_G)
        self.add_module('model_D', model_D)

        self.add_module('l1_loss', l1_loss)
        self.add_module('l2_loss', l2_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # optimizer
        self.optimizer_G = torch.optim.Adam(params=self.model_G.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(params=self.model_D.parameters(),
                                            lr=args.lr * args.d2g_lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))

        if not args.use_canny:
            stru_net_skip_conn = [int(args.stru_net_skip_conn[i]) for i in range(len(args.stru_net_skip_conn))]
            model_Stru = Controllable_UNet(3, 2, stru_net_skip_conn, unit_channel=args.stru_unit_channel)
            model_Stru = nn.DataParallel(model_Stru).cuda()
            self.add_module('model_Stru', model_Stru)

            stru_ckpt_root = os.path.join(self.args.output_root, self.args.Stru_load_version, 'checkpoints')
            ckpt_path = os.path.join(stru_ckpt_root, args.Stru_resume)
            if os.path.isfile(ckpt_path):
                print("=> loading Stru Net checkpoint '{}'".format(args.Stru_resume))
                checkpoint = torch.load(ckpt_path)
                self.model_Stru.load_state_dict(checkpoint['state_dict_Stru'])
                print("=> loaded Stru Net checkpoint '{}' (epoch {})"
                      .format(args.Stru_resume, checkpoint['epoch']))
            else:
                print("=> no Stru Net checkpoint found at '{}'".format(args.Stru_resume))

        if self.args.resume:
            ckpt_root = os.path.join(self.args.output_root, self.args.version, 'checkpoints')
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading G checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                self.model_G.load_state_dict(checkpoint['state_dict_G'])
                print("=> loaded G checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def process(self, image):
        # process_outputs
        edge, image_rec = self(image)

        image = image_blurr(image, kernel_size=self.args.gau_kernel,
                            sigma=self.args.gau_sigma, convert=self.args.image_mode)
        """
        G and D process, this package is reusable
        """
        # zero optimizers
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        gen_loss = 0
        dis_loss = 0

        real_B = image.cuda()
        fake_B = image_rec

        # discriminator loss
        dis_input_real = real_B
        dis_input_fake = fake_B.detach()
        dis_real, dis_real_feat = self.model_D(dis_input_real)
        dis_fake, dis_fake_feat = self.model_D(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = fake_B
        gen_fake, gen_fake_feat = self.model_D(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.args.lamd_gen
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l2_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.args.lamd_fm
        gen_loss += gen_fm_loss

        # # generator l1 loss
        # gen_l1_loss = self.l1_loss(fake_B, real_B) * self.args.lamd_p
        # gen_loss += gen_l1_loss

        # generator l2 loss
        gen_l2_loss = self.l2_loss(fake_B, real_B) * self.args.lamd_p
        gen_loss += gen_l2_loss
        """
        VGG loss, this package is reusable
        """

        # create logs
        logs = dict(
            gen_gan_loss=gen_gan_loss,
            gen_fm_loss=gen_fm_loss,
            gen_l2_loss=gen_l2_loss,
        )

        return edge, fake_B, gen_loss, dis_loss, logs

    def forward(self, image):
        with torch.no_grad():
            if self.args.use_canny:
                edge = canny(image, self.args.canny_sigma)
            else:
                edge_map = self.model_Stru(image)
                edge = edge_map.max(dim=1)[1].unsqueeze(1).float()
        image_rec = self.model_G(image, edge)

        return edge, image_rec

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.optimizer_D.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.optimizer_G.step()


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
                             crop_rate=args.crop_rate,
                             flip_rate=args.flip_rate).data_load()

        print_args(args)
        self.epoch = args.start_epoch
        self.args = args
        self.new_lr = self.args.lr
        self.model = PNetModel(args)

        self.best_auc = 0
        self.best_acc = 0
        self.best_iou = 0
        self.best_overlap = 0
        self.is_best = False
        self.auc_top20 = AverageMeter()
        self.auc_last20 = LastAvgMeter(length=20)
        self.iou_top20 = AverageMeter()
        self.iou_last20 = LastAvgMeter(length=20)
        self.overlap_top20 = AverageMeter()
        self.overlap_last20 = LastAvgMeter(length=20)
        self.threshold = 0
        if args.predict:
            self.validate_cls()
        else:
            self.train_val()

    def train_val(self):
        # general metrics
        self.vis.text(str(vars(self.args)), name='args')
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            # adjust_lr_epoch_list = [int(self.args.n_epochs * 0.5), int(self.args.n_epochs * 0.75)]
            adjust_lr_epoch_list = [int(self.args.n_epochs * 0.7), int(self.args.n_epochs * 0.9)]
            _ = adjust_lr(self.args.lr, self.model.optimizer_G, epoch, adjust_lr_epoch_list)
            _ = adjust_lr(self.args.lr * self.args.d2g_lr, self.model.optimizer_D, epoch, adjust_lr_epoch_list)

            self.epoch = epoch
            self.train()

            if (epoch + 1) % self.args.val_freq == 0 or epoch == 0:
                self.validate_cls()

            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('Version: {}\n'.format(self.args.version))

        self.vis.save(self.args.version)

    def train(self):
        self.model.train()
        prev_time = time.time()
        train_loader = self.normal_train_loader
        for i, (image, _, _) in enumerate(train_loader):
            image = image.cuda(non_blocking=True)
            # train
            edge, image_rec, gen_loss, dis_loss, logs = \
                self.model.process(image)
            image = image_blurr(image, kernel_size=self.args.gau_kernel,
                                sigma=self.args.gau_sigma, convert=self.args.image_mode).cuda()
            # backward
            self.model.backward(gen_loss, dis_loss)
            # --------------
            #  Log Progress
            # --------------
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
                image_rec = image_rec[:self.args.vis_batch]
                image_diff = torch.abs(image-image_rec)
                edge_vis = torch.cat([edge, edge, edge], dim=1)
                vim_images = torch.cat([image, edge_vis.cuda(), image_rec, torch.clamp(image_diff * 3, 0, 1)], dim=0)
                self.vis.images(vim_images, win_name='train', nrow=self.args.vis_batch)

            if i+1 == train_loader.__len__():
                self.vis.plot_multi_win(dict(dis_loss=dis_loss.item()))
                self.vis.plot_single_win(dict(gen_l2_loss=logs['gen_l2_loss'].item(),
                                              gen_fm_loss=logs['gen_fm_loss'].item(),
                                              gen_gan_loss=logs['gen_gan_loss'].item()),
                                         win='gen_loss')

    def validate_cls(self):
        self.model.eval()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            _, normal_train_pred_list, _, _ = self.forward_cls_dataloader(
                loader=self.normal_train_loader, is_disease=False)
            normal_test_gt_list, normal_test_pred_list, _, _ = self.forward_cls_dataloader(
                loader=self.normal_test_loader, is_disease=False, category='val_normal')
            abnormal_gt_list, abnormal_pred_list, abnormal_iou, abnormal_overlap = self.forward_cls_dataloader(
                loader=self.abnormal_loader, is_disease=True, category='val_abnormal')

            """
            computer metrics
            """
            auc_list = []
            true_list = abnormal_gt_list + normal_test_gt_list
            for p in range(len(normal_test_pred_list)):
                pred_list = abnormal_pred_list[p] + normal_test_pred_list[p]

                # get roc curve and compute the auc
                fpr, tpr, thresholds = metrics.roc_curve(np.array(true_list), np.array(pred_list))
                auc_list.append(metrics.auc(fpr, tpr))
            auc = auc_list[3]

            # compute iou
            iou = torch.mean(abnormal_iou).item()
            overlap = torch.mean(abnormal_overlap).item()

            # update
            self.auc_last20.update(auc)
            # self.acc_last20.update(acc)
            self.iou_last20.update(iou)
            self.overlap_last20.update(overlap)

            self.is_best = auc > self.best_auc
            self.best_auc = max(auc, self.best_auc)
            # self.best_acc = max(acc, self.best_acc)
            self.best_iou = max(iou, self.best_iou)
            self.best_overlap = max(overlap, self.best_overlap)

            """
            plot metrics curve
            """
            # ROC curve
            self.vis.draw_roc(fpr, tpr)
            # total auc, primary metrics
            self.vis.plot_single_win(dict(value=auc,
                                          # value_1=auc_list[4],
                                          # value_2=auc_list[5],
                                          # value_3=auc_list[6],
                                          # value_m1=auc_list[2],
                                          # value_m2=auc_list[1],
                                          # value_m3=auc_list[0],
                                          ), win='auc')
            self.vis.plot_single_win(dict(value=iou,
                                          best=self.best_iou,
                                          last_avg=self.iou_last20.avg,
                                          last_std=self.iou_last20.std,
                                          ), win='iou')
            self.vis.plot_single_win(dict(value=overlap,
                                          best=self.best_overlap,
                                          last_avg=self.overlap_last20.avg,
                                          last_std=self.overlap_last20.std,
                                          ), win='overlap')
            # self.vis.plot_single_win(dict(train_normal=torch.mean(torch.FloatTensor(normal_train_pred_list)),
            #                               test_normal=torch.mean(torch.FloatTensor(normal_test_pred_list)),
            #                               test_abnormal=torch.mean(torch.FloatTensor(abnormal_pred_list))),
            #                          win='pred_cost')

            metrics_str = 'best_auc = {:.4f},' \
                          'auc_last20_avg = {:.4f}, auc_last20_std = {:.4f}, '.\
                format(self.best_auc, self.auc_last20.avg, self.auc_last20.std)
            metrics_overlap_str = '\n best_overlap = {:.4f}, overlap_last20_avg = {:.4f}, threshold:{:.4f}'.\
                format(self.best_overlap, self.overlap_last20.avg, self.threshold)

            self.vis.text(metrics_str + metrics_overlap_str, name='text')

        save_ckpt(version=self.args.version,
                  state={
                      'epoch': self.epoch,
                      'state_dict_G': self.model.model_G.state_dict(),
                      'state_dict_D': self.model.model_D.state_dict(),
                  },
                  epoch=self.epoch,
                  is_best=self.is_best,
                  args=self.args)

        print('\n Save ckpt successfully!')
        print('\n', metrics_str + metrics_overlap_str)

    def detect_iou(self):
        self.model.train()
        iou_dict = {}
        threshold_list = np.linspace(0, 0.4, 100)
        for threshold in threshold_list:
            iou_dict[threshold] = AverageMeter()
        for i, (image, image_name_item, mask) in enumerate(self.abnormal_loader):
            image = image.cuda(non_blocking=True)
            # val, forward
            edge, image_rec = self.model(image)
            if self.args.gau_sigma != 0:
                image = image_blurr(image, kernel_size=self.args.gau_kernel,
                                    sigma=self.args.gau_sigma, convert=self.args.image_mode)
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

    def forward_cls_dataloader(self, loader, is_disease, category='train_normal'):
        gt_list = []
        pred_list = [[], [], [], [], [], [], []]
        iou_list = []
        overlap_list = []
        threshold_sum = AverageMeter()
        for i, (image, image_name_item, mask) in enumerate(loader):
            mask = mask.cuda()
            image = image.cuda(non_blocking=True)
            # val, forward
            edge, image_rec = self.model(image)
            image_name = image_name_item
            if self.args.gau_sigma != 0:
                image = image_blurr(image, kernel_size=self.args.gau_kernel,
                                    sigma=self.args.gau_sigma, convert=self.args.image_mode).cuda()

            """
            preditction
            """
            # use args.pixpow to make anomaly region more saliency
            image_diff = torch.abs(image_rec - image)
            image_diff_cut = diff_cut(image_diff, self.args.cut_rate)
            for p in [-3, -2, -1, 0, 1, 2, 3]:
                image_diff_mean = (image_diff_cut ** (self.args.pixpow + p)).mean(dim=3).mean(dim=2).mean(dim=1)
                pred_list[p+3] += image_diff_mean.tolist()
            gt_list += [1 if is_disease else 0] * len(image_name)
            """
            segmentation threshold
            """
            if category == 'val_normal':
                for im in range(image.size(0)):
                    diff_list = torch.sort(image_diff[im].view(-1))
                    threshold_sum.update(diff_list[0][int(self.args.th_rate * diff_list[1].max().float())].item())
                self.threshold = threshold_sum.avg
            elif category == 'val_abnormal':
                ano_region_mask = (image_diff.mean(dim=1) >= self.threshold).float()
                # ano_region_mask = dilated_eroded(ano_region_mask)
                iou_list += [torch.sum(ano_region_mask[j] * mask[j, 0]) /
                             torch.sum((ano_region_mask[j] + mask[j, 0] > 0).float()) for j in range(image.size(0))]
                overlap_list += [torch.sum(ano_region_mask[j] * mask[j, 0]) /
                                 torch.sum((mask[j, 0] > 0).float()) for j in range(image.size(0))]
            """
            save images
            """
            if (self.epoch + 1) % self.args.save_image_freq == 0 or self.args.predict:
                edge_save = torch.cat([edge, edge, edge], dim=1).cuda()
                if category == 'val_abnormal':
                    ano_region_mask = (image_diff.mean(dim=1) >= self.threshold).float()
                    region_mask_pred = torch.cat([ano_region_mask.unsqueeze(1)] * 3, dim=1)
                    mask_vis = torch.cat([mask, mask, mask], dim=1)
                    vim_images = torch.cat([image, edge_save.cuda(), image_rec, torch.clamp(image_diff * 3, 0, 1),
                                            region_mask_pred.cuda(), mask_vis.cuda()],
                                           dim=0)
                else:
                    vim_images = torch.cat([image, edge_save.cuda(), image_rec, torch.clamp(image_diff * 3, 0, 1)], dim=0)

                output_save = os.path.join(self.args.output_root,
                                           '{}'.format(self.args.version),
                                           'sample')

                if not os.path.exists(output_save):
                    os.makedirs(output_save)
                print('saving images:[{}/{}]'.format(i, len(loader)))
                tv.utils.save_image(vim_images, os.path.join(
                    output_save, '{}_{}_{}.png'.format(category, self.epoch, i)), nrow=image.size(0))

            if category != 'train_normal' and i == 0:
                """
                visdom
                """
                image = image[:self.args.vis_batch]
                image_rec = image_rec[:self.args.vis_batch]
                mask = mask[:self.args.vis_batch]
                image_diff = torch.abs(image - image_rec)

                edge = edge[:self.args.vis_batch]
                edge_vis = torch.cat([edge, edge, edge], dim=1).cuda()
                if category == 'val_normal':

                    vim_images = torch.cat([image,
                                            edge_vis.cuda(),
                                            image_rec,
                                            torch.clamp(image_diff * 3, 0, 1)],
                                           dim=0)
                else:
                    ano_region_mask = (image_diff.mean(dim=1) >= self.threshold).float()
                    region_mask_pred = torch.cat([ano_region_mask.unsqueeze(1)] * 3, dim=1)
                    region_mask_pred_vis = region_mask_pred[:self.args.vis_batch]

                    mask_vis = torch.cat([mask, mask, mask], dim=1)
                    vim_images = torch.cat([image, edge_vis.cuda(),
                                            image_rec,
                                            torch.clamp(image_diff * 3, 0, 1),
                                            region_mask_pred_vis.cuda(),
                                            mask_vis.cuda()],
                                           dim=0)
                self.vis.images(vim_images, win_name='{}'.format(category), nrow=self.args.vis_batch)

        return gt_list, pred_list, torch.FloatTensor(iou_list), torch.FloatTensor(overlap_list)


if __name__ == '__main__':
    import pdb
    RunMyModel()
    # MultiTestForFigures()
