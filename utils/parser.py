# -*- coding: utf-8 -*-
import argparse
import warnings
import os


class ParserArgs(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='PyTorch Training and Testing'
        )

        # self.parser addition
        self.constant_init()
        self.get_general_parser()
        self.get_network_parser()
        self.get_data_parser()
        self.get_parms_parser()
        self.get_freq_and_other_parser()

        # ablation exps
        self.get_ablation_exp_args()
        # comparison exps
        self.get_comparison_exps_args()

    def get_network_parser(self):
        self.parser.add_argument('--stru_net_skip_conn', default=[1, 1, 1, 1], type=list, help='project name in workspace')
        self.parser.add_argument('--stru_unit_channel', default=8, type=int, help='project name in workspace')
        self.parser.add_argument('--image_skip_conn', default=[0, 0, 0], type=list, help='project name in workspace')

    def constant_init(self):
        # path and server
        self.data_root = '/p300/datasets'
        self.output_root = '/root/ECCV-2020/workspace'
        self.vis_server = 'http://10.10.10.100'
        # constant
        self.parser.add_argument('--project', default='ano_det_cvpr2020',
                                 help='project name in workspace')
        self.parser.add_argument('--d2g_lr', type=float, default=0.1,
                                 help='discriminator/generator learning rate ratio')
        self.parser.add_argument('--weight_decay', default=1e-4, type=float,
                                 metavar='W', help='weight decay (default: 1e-4)')
        self.parser.add_argument('--canny_sigma', default=1, type=float, help='canny sigma of Stru Net learning')
        self.parser.add_argument('--image_mode', default='RGB', type=str, help='L or RGB')

    def get_general_parser(self):
        # general useful args
        self.parser.add_argument('--version', default='v99_debug',
                                 help='the version of different method/setting/parameters etc')
        self.parser.add_argument('--Stru_load_version', default='', type=str,
                                 help='the version of Stru Net')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                 help='path to latest checkpoint ')
        self.parser.add_argument('--Stru_resume', default='', type=str, metavar='PATH',
                                 help='path to latest checkpoint of Stru(format: version/checkpoints/path.tar)')
        self.parser.add_argument('--test', action='store_true',
                                 help='test mode, rather than train and val')
        self.parser.add_argument('--use_canny', action='store_true',
                                 help='Use canny detector for edge segmentation in 1st stage')
        self.parser.add_argument('--port', default=31670, type=int, help='visdom port')
        self.parser.add_argument('--gpu', nargs='+', type=int,
                                 help='gpu id for single/multi gpu')

    def get_data_parser(self):
        # dataset
        self.parser.add_argument('--data_modality',
                                 choices=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                                          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                                          'wood', 'zipper'],
                                 help='the modality of data. No default.')
        self.parser.add_argument('--scale', default=224, type=int,
                                 help='image scale (default: 224)')

        # # data augmentation
        self.parser.add_argument('--enhance_p', type=float, default=0)
        self.parser.add_argument('--flip', type=bool, default=False)
        self.parser.add_argument('--rotate', type=int, default=0)
        self.parser.add_argument('--crop_size', type=int, default=700)
        self.parser.add_argument('--crop_rate', type=float, default=0)
        self.parser.add_argument('--flip_rate', type=float, default=0)

    def get_parms_parser(self):
        # model hyper-parameters
        self.parser.add_argument('--start_epoch', default=0, type=int,
                                 help='numbet of start epoch of edge segmentation to run')
        self.parser.add_argument('--n_epochs', default=800, type=int, metavar='N',
                                 help='number of total epochs to run')
        self.parser.add_argument('--batch', default=18, type=int,
                                 metavar='N', help='mini-batch size')
        self.parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                                 metavar='LR', help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                 help='momentum')

        # pixel gan
        self.parser.add_argument('--b1', type=float, default=0.1,
                                 help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--b2', type=float, default=0.9,
                                 help='adam: decay of first order momentum of gradient')
        self.parser.add_argument('--lamd_p', '--lamd_pixel', default=1, type=float,
                                 help='Loss weight of L1 pixel-wise loss between translated image and real image')
        self.parser.add_argument('--lamd_fm', default=0.01, type=float)
        self.parser.add_argument('--lamd_gen', default=0.1, type=float)
        self.parser.add_argument('--latent_size', default=1024, type=int)

        # # iou threshold
        self.parser.add_argument('--th_rate', type=float, default=0.98)

        self.parser.add_argument('--gau_kernel', type=int, default=11, help='size of gaussian kernel')
        self.parser.add_argument('--gau_sigma', type=float, default=2.5, help='sigma of gaussian')

        self.parser.add_argument('--pixpow', type=float, default=1, help='pixel pow')
        self.parser.add_argument('--cut_rate', type=float, default=0, help='pixel cut rate')
        self.parser.add_argument('--diff_th', type=float, default=0, help='diff threshold')

    def get_ablation_exp_args(self):
        # multi-level network
        self.parser.add_argument('--ablation_mode', default=4, choices=[0, 1, 2, 3, 4, 5, 6],
                                 type=int, help='ablation study for multi-level feature')
        # 3nd-stage, residual fusion
        self.parser.add_argument('--lamd_mask_fusion', default=0, type=float,)
        # cycle mask loss
        self.parser.add_argument('--stage3_epoch', default=0, type=int)
        self.parser.add_argument('--lamd_mask', default=10, type=float,
                                 help='range = (0, 1)')
        self.parser.add_argument('--lamd_edge', default=0.1, type=int, help='cycle edge loss')
        # ganomaly latent_l1 loss
        self.parser.add_argument('--lamd_lat', '--lamd_latent_l1', default=1, type=float)

    def get_comparison_exps_args(self):
        self.parser.add_argument('--com_model_name', choices=['ae', 'ae_gan', 'pix_gan', 'ganomaly',
                                                              'cycle_gan'], type=str,
                                 help='the comparison exps name')
        self.parser.add_argument('--has_D', default=False, action='store_true',)

    # using this function when define the obeject of the class
    def get_args(self):
        crop_size = {'carpet': 800, 'grid': 800, 'leather': 800, 'tile': 650, 'wood': 800, 'bottle': 700,
                     'cable': 800, 'capsule': 800, 'hazelnut': 800, 'metal_nut': 550, 'pill': 650, 'screw': 800,
                     'toothbrush': 800, 'transistor': 800, 'zipper': 800}
        args = self.parser.parse_args()
        # xiaoyt-XX
        args.mvtec_root = os.path.join(self.data_root, 'mvtec_anomaly_detection')
        args.load_Stru_path = '/root/ECCV-2020/workspace/{}/Stru_latest_ckpt.pth.tar'.format(args.Stru_load_version)
        args.vis_server = self.vis_server
        args.output_root = self.output_root

        args.crop_size = crop_size[args.data_modality]

        self.assert_version(args.version)

        return args

    def get_freq_and_other_parser(self):
        # other useful args
        self.parser.add_argument('--val_freq', default=10, type=int,
                                 help='validate frequency (default: 5)')
        self.parser.add_argument('--val_start_epoch', default=0, type=int)
        self.parser.add_argument('--save_model_freq', default=500, type=int)
        self.parser.add_argument('--save_image_freq', default=150, type=int)
        self.parser.add_argument('--vis_batch', default=10, type=int)
        self.parser.add_argument('--print_freq', default=90, type=int,
                                 metavar='N', help='print frequency (default: 90)')

    @staticmethod
    def assert_version(version):
        # format: v01_XXX&XXX&sss_XXX
        v_split_list = version.split('_')
        v_major = v_split_list[0][0] == 'v' and v_split_list[0][1:].isdigit() and len(v_split_list[0]) == 3
        if not v_major:
            warnings.warn('The version name is warning')
