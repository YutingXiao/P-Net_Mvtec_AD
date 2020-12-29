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
from networks.AE_GAN import Encoder, Decoder
from networks.discriminator import Discriminator
from utils.vgg_loss import AdversarialLoss, PerceptualLoss, StyleLoss
from utils.visualizer import Visualizer
from utils.trick import adjust_lr, cuda_visible, print_args, save_ckpt, AverageMeter, LastAvgMeter
from utils.parser import ParserArgs


class AE_NetModel(nn.Module):
    def __init__(self, args):
        super(AE_NetModel, self).__init__()
        self.args = args
        model_E = Encoder(args.latent_size, channel=3, mode='dae')
        model_De = Decoder(args.latent_size, output_channel=3, mode='dae')
        model_D = Discriminator(in_channels=3)

        model_E = nn.DataParallel(model_E).cuda()
        model_De = nn.DataParallel(model_De).cuda()
        model_D = nn.DataParallel(model_D).cuda()

        l1_loss = nn.L1Loss().cuda()
        l2_loss = nn.MSELoss().cuda()
        adversarial_loss = AdversarialLoss().cuda()

        # self.add_module('model_G1', model_G1)
        self.add_module('model_E', model_E)
        self.add_module('model_De', model_De)
        self.add_module('model_D', model_D)

        self.add_module('l1_loss', l1_loss)
        self.add_module('l2_loss', l2_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # optimizer
        self.optimizer_E = torch.optim.Adam(params=self.model_E.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))
        self.optimizer_De = torch.optim.Adam(params=self.model_De.parameters(),
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(params=self.model_D.parameters(),
                                            lr=args.lr * args.d2g_lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))

        if self.args.resume:
            ckpt_root = os.path.join(self.args.output_root, self.args.version, 'checkpoints')
            ckpt_path = os.path.join(ckpt_root, args.resume)
            if os.path.isfile(ckpt_path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                self.model_E.load_state_dict(checkpoint['state_dict_E'])
                self.model_De.load_state_dict(checkpoint['state_dict_De'])
                self.model_D.load_state_dict(checkpoint['state_dict_D'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def process(self, image):
        # process_outputs
        image_rec = self(image)


        """
        G and D process, this package is reusable
        """
        # zero optimizers
        self.optimizer_E.zero_grad()
        self.optimizer_De.zero_grad()
        self.optimizer_D.zero_grad()

        gen_loss = 0
        dis_loss = 0

        real_B = image
        fake_B = image_rec

        # discriminator loss
        dis_input_real = real_B
        dis_input_fake = fake_B.detach()
        dis_real, dis_real_feat = self.model_D(dis_input_real)
        dis_fake, dis_fake_feat = self.model_D(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += self.args.lamd_gen * (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = fake_B
        gen_fake, gen_fake_feat = self.model_D(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.args.lamd_gen
        gen_loss += gen_gan_loss

        # generator l2 loss
        gen_l2_loss = self.l2_loss(fake_B, real_B) * self.args.lamd_p
        gen_loss += gen_l2_loss

        # create logs
        logs = dict(
            gen_gan_loss=gen_gan_loss,
            gen_l2_loss=gen_l2_loss,
        )

        return fake_B, gen_loss, dis_loss, logs

    def forward(self, image):
        z = self.model_E(image)
        image_rec = self.model_De(z)

        return image_rec

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.optimizer_D.step()

        if gen_loss is not None:
            gen_loss.backward()
        self.optimizer_De.step()
        self.optimizer_E.step()


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
        self.model = AE_NetModel(args)

        self.threshold = 0.1

        self.best_auc = 0
        self.best_acc = 0
        self.best_iou = 0
        self.is_best = False
        self.auc_top10 = AverageMeter()
        self.auc_last20 = LastAvgMeter()
        self.acc_top10 = AverageMeter()
        self.acc_last20 = LastAvgMeter()
        self.iou_top10 = AverageMeter()
        self.iou_last20 = LastAvgMeter()

        if args.predict:
            pass
        else:
            self.train_val()

    def train_val(self):
        # general metrics
        self.vis.text(str(vars(self.args)), name='args')
        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            adjust_lr_epoch_list = [500]
            _ = adjust_lr(self.args.lr, self.model.optimizer_E, epoch, adjust_lr_epoch_list)
            _ = adjust_lr(self.args.lr, self.model.optimizer_De, epoch, adjust_lr_epoch_list)
            _ = adjust_lr(self.args.lr * self.args.d2g_lr, self.model.optimizer_D, epoch, adjust_lr_epoch_list)

            self.epoch = epoch
            self.train(epoch)

            if epoch % self.args.val_freq == 0:
                self.validate_cls()

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
            image_rec, gen_loss, dis_loss, logs = self.model.process(image)

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
                image_rec = image_rec[:self.args.vis_batch]
                image_diff = torch.abs(image-image_rec)
                vim_images = torch.cat([image, image_rec, image_diff], dim=0)
                self.vis.images(vim_images, win_name='train', nrow=self.args.vis_batch)

                output_save = os.path.join(self.args.output_root,
                                           self.args.version,
                                           'sample')
                os.makedirs(output_save, exist_ok=True)
                tv.utils.save_image(vim_images,
                                    os.path.join(output_save, 'train_{}_{}.png'.format(epoch, i)),
                                    nrow=self.args.vis_batch)

            if i+1 == train_loader.__len__():
                self.vis.plot_multi_win(dict(dis_loss=dis_loss.item()))
                self.vis.plot_single_win(dict(gen_loss=gen_loss.item(),
                                              gen_l2_loss=logs['gen_l2_loss'].item(),
                                              gen_gan_loss=logs['gen_gan_loss'].item()),
                                         win='gen_loss')

    def validate_cls(self):
        self.model.eval()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            _, normal_train_pred_list, _ = self.forward_cls_dataloader(
                loader=self.normal_train_loader, is_disease=False)
            abnormal_gt_list, abnormal_pred_list, abnormal_iou = self.forward_cls_dataloader(
                loader=self.abnormal_loader, is_disease=True, category='val_abnormal')
            normal_test_gt_list, normal_test_pred_list, _ = self.forward_cls_dataloader(
                loader=self.normal_test_loader, is_disease=False, category='val_normal')

            """
            computer metrics
            """
            # Difference: total_true_list and total_pred_list
            # test metrics for myopia
            true_list = abnormal_gt_list + normal_test_gt_list
            pred_list = abnormal_pred_list + normal_test_pred_list

            # get roc curve and compute the auc
            fpr, tpr, thresholds = metrics.roc_curve(np.array(true_list), np.array(pred_list))
            auc = metrics.auc(fpr, tpr)

            """
            compute thereshold, and then compute the accuracy
            """
            percentage = 0.75
            threshold_for_acc = sorted(normal_train_pred_list)[int(len(normal_train_pred_list) * percentage)]
            normal_cls_pred_list = [(0 if i < threshold_for_acc else 1) for i in normal_test_pred_list]
            abnormal_cls_pred_list = [(0 if i < threshold_for_acc else 1) for i in abnormal_pred_list]

            # acc, sensitivity and specifity
            cls_pred_list = normal_cls_pred_list + abnormal_cls_pred_list
            gt_list = normal_test_gt_list + abnormal_gt_list
            acc = metrics.accuracy_score(y_true=gt_list, y_pred=cls_pred_list)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true=gt_list, y_pred=cls_pred_list).ravel()
            sen = tp / (tp + fn + 1e-7)
            spe = tn / (tn + fp + 1e-7)

            # compute iou
            iou = torch.mean(abnormal_iou)

            # update
            self.auc_last20.update(auc)
            self.acc_last20.update(acc)
            self.iou_last20.update(iou)

            auc_mean, auc_deviation = self.auc_top10.top_update_calc(auc)
            acc_mean, acc_deviation = self.acc_top10.top_update_calc(acc)
            iou_mean, iou_deviation = self.iou_top10.top_update_calc(iou)

            self.is_best = auc > self.best_auc
            self.best_auc = max(auc, self.best_auc)
            self.best_acc = max(acc, self.best_acc)
            self.best_iou = max(iou, self.best_iou)

            """
            plot metrics curve
            """
            # ROC curve
            self.vis.draw_roc(fpr, tpr)
            # total auc, primary metrics
            self.vis.plot_single_win(dict(value=auc,
                                          best=self.best_auc,
                                          last_avg=self.auc_last20.avg,
                                          last_std=self.auc_last20.std,
                                          top_avg=auc_mean,
                                          top_dev=auc_deviation), win='auc')
            self.vis.plot_single_win(dict(value=acc,
                                          best=self.best_acc,
                                          last_avg=self.acc_last20.avg,
                                          last_std=self.acc_last20.std,
                                          top_avg=acc_mean,
                                          top_dev=acc_deviation,
                                          sen=sen,
                                          spe=spe), win='accuracy')
            self.vis.plot_single_win(dict(value=iou,
                                          best=self.best_iou,
                                          last_avg=self.iou_last20.avg,
                                          last_std=self.iou_last20.std,
                                          top_avg=iou_mean,
                                          top_dev=iou_deviation), win='iou')

            metrics_str = 'best_auc = {:.4f},' \
                          'auc_last20_avg = {:.4f}, auc_last20_std = {:.4f}, ' \
                          'auc_top10_avg = {:.4f}, auc_top10_dev = {:.4f}, '.\
                format(self.best_auc, self.auc_last20.avg, self.auc_last20.std, auc_mean, auc_deviation)
            metrics_acc_str = '\n best_acc = {:.4f}, ' \
                              'acc_last20_avg = {:.4f}, acc_last20_std = {:.4f}, ' \
                              'acc_top10_avg = {:.4f}, acc_top10_dev = {:.4f}, '.\
                format(self.best_acc, self.acc_last20.avg, self.acc_last20.std, acc_mean, acc_deviation)

            self.vis.text(metrics_str + metrics_acc_str)

        save_ckpt(version=self.args.version,
                  state={
                      'epoch': self.epoch,
                      'state_dict_E': self.model.model_E.state_dict(),
                      'state_dict_De': self.model.model_De.state_dict(),
                      'state_dict_D': self.model.model_D.state_dict(),
                  },
                  epoch=self.epoch,
                  is_best=self.is_best,
                  args=self.args)

        print('\n Save ckpt successfully!')
        print('\n', metrics_str + metrics_acc_str)

    def forward_cls_dataloader(self, loader, is_disease, category='train_normal'):
        gt_list = []
        pred_list = []
        iou_list = []
        threshold = self.threshold
        for i, (image, image_name_item, mask) in enumerate(loader):
            mask = mask.cuda()
            image = image.cuda(non_blocking=True)
            # val, forward
            image_rec = self.model(image)
            image_name = image_name_item
            """
            val loss
            """
            if category == 'val_normal':
                loss = torch.sum((image - image_rec) ** 2) / (image.size(0) * image.size(1) * image.size(2) * image.size(3))
                self.vis.plot_single_win(dict(val_l2_loss=loss), win='val_loss')


            """
            preditction
            """
            # BCWH -> B, anomaly score
            image_diff = torch.abs(image_rec - image)
            image_diff_mean = image_diff.mean(dim=3).mean(dim=2).mean(dim=1)
            gt_list += [1 if is_disease else 0] * len(image_name)
            pred_list += image_diff_mean.tolist()

            if category == 'val_abnormal':
                mask = mask.cuda()
                ano_region_mask = (image_diff.mean(dim=1) >= threshold).float()
                iou_list += [torch.sum(ano_region_mask[j] * mask[j, 0]) /
                             torch.sum(((ano_region_mask[j] + mask[j, 0]) > 0).float()) for j in range(image.size(0))]
                mask_vis = torch.cat([mask, mask, mask], dim=1)
                vim_images = torch.cat([image, image_rec, image_diff, mask_vis.cuda()], dim=0)
            else:
                vim_images = torch.cat([image, image_rec, image_diff], dim=0)

            """
            save images
            """

            output_save = os.path.join(self.args.output_root,
                                       '{}'.format(self.args.version),
                                       'sample')

            if not os.path.exists(output_save):
                os.makedirs(output_save)
            tv.utils.save_image(vim_images, os.path.join(
                output_save, '{}_{}_{}.png'.format(category, self.epoch, i)), nrow=image.size(0))
            """
            visdom
            """
            if i == 0 and category != 'train_normal':
                image = image[:self.args.vis_batch]
                image_rec = image_rec[:self.args.vis_batch]
                mask = mask[:self.args.vis_batch]
                image_diff = torch.abs(image - image_rec)
                """
                Difference: edge is different between fundus and oct images
                """
                if category == 'val_normal':
                    vim_images = torch.cat([image, image_rec, image_diff], dim=0)
                else:
                    mask_vis = torch.cat([mask, mask, mask], dim=1)
                    vim_images = torch.cat([image, image_rec, image_diff, mask_vis.cuda()], dim=0)

                self.vis.images(vim_images, win_name='{}'.format(category), nrow=self.args.vis_batch)

                for n in range(self.args.vis_batch):
                    self.vis.plot_histogram(image_diff[n].max(dim=1)[0].view(-1), win='{}_{}'.format(category, n), numbins=500)
                    if n > 4:
                        break
        return gt_list, pred_list, torch.FloatTensor(iou_list)


if __name__ == '__main__':
    import pdb
    RunMyModel()
    # MultiTestForFigures()
