import torch
import torch.nn as nn
import pdb


class Controllable_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, skip_conn=[0, 0, 0, 0], unit_channel=10, last_layer='sigmoid'):
        super(Controllable_UNet, self).__init__()
        self.last_layer = last_layer
        feature_channel = [unit_channel * 2, unit_channel * 4, unit_channel * 8, unit_channel * 16]
        for i in range(len(skip_conn)):
            if skip_conn[i] == 1:
                feature_channel[i] += int(feature_channel[i] * 0.5)

        self.inc = inconv(n_channels, unit_channel)
        self.down1 = down(unit_channel, unit_channel * 2)
        self.down2 = down(unit_channel * 2, unit_channel * 4)
        self.down3 = down(unit_channel * 4, unit_channel * 8)
        self.down4 = down(unit_channel * 8, unit_channel * 16)

        self.up1 = up(feature_channel[3], unit_channel * 8)
        self.up2 = up(feature_channel[2], unit_channel * 4)
        self.up3 = up(feature_channel[1], unit_channel * 2)
        self.up4 = up(feature_channel[0], unit_channel)
        self.outc = outconv(unit_channel, n_classes, self.last_layer)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class Dual_Path_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, blurr_skip_conn=[0, 0, 0, 0], edge_skip_conn=[0, 0, 0, 0],
                 unit_channel=16, last_layer='sigmoid'):
        super(Dual_Path_UNet, self).__init__()
        self.last_layer = last_layer
        feature_channel = [unit_channel * 2, unit_channel * 4, unit_channel * 8, unit_channel * 16]
        for i in range(len(blurr_skip_conn)):
            feature_channel[i] += int(feature_channel[i] * (blurr_skip_conn[i] + edge_skip_conn[i]) * 0.5)

        self.inc = inconv(n_channels, unit_channel)
        self.blurr_down1 = down(unit_channel, unit_channel * 2)
        self.blurr_down2 = down(unit_channel * 2, unit_channel * 4)
        self.blurr_down3 = down(unit_channel * 4, unit_channel * 8)
        self.blurr_down4 = down(unit_channel * 8, unit_channel * 16)

        self.edge_down1 = down(unit_channel, unit_channel * 2)
        self.edge_down2 = down(unit_channel * 2, unit_channel * 4)
        self.edge_down3 = down(unit_channel * 4, unit_channel * 8)
        self.edge_down4 = down(unit_channel * 8, unit_channel * 16)

        self.up1 = dual_up(feature_channel[3], unit_channel * 8, [blurr_skip_conn[3], edge_skip_conn[3]])
        self.up2 = dual_up(feature_channel[2], unit_channel * 4, [blurr_skip_conn[2], edge_skip_conn[2]])
        self.up3 = dual_up(feature_channel[1], unit_channel * 2, [blurr_skip_conn[1], edge_skip_conn[1]])
        self.up4 = dual_up(feature_channel[0], unit_channel, [blurr_skip_conn[0], edge_skip_conn[0]])
        self.outc = outconv(unit_channel, n_classes, self.last_layer)

    def forward(self, x1, x2):
        x11 = self.inc(x1)             # x11: unit   *   224   *   224
        x12 = self.blurr_down1(x11)    # x12: 2unit  *   112   *   112
        x13 = self.blurr_down2(x12)    # x13: 4unit  *   56    *   56
        x14 = self.blurr_down3(x13)    # x14: 8unit  *   28    *   28
        x15 = self.blurr_down4(x14)    # x15: 16unit *   14    *   14

        x21 = self.inc(x2)
        x22 = self.edge_down1(x21)
        x23 = self.edge_down2(x22)
        x24 = self.edge_down3(x23)
        x25 = self.edge_down4(x24)

        x = torch.cat([x15, x25], dim=1)
        x = self.up1(x, x14, x24)
        x = self.up2(x, x13, x23)
        x = self.up3(x, x12, x22)
        x = self.up4(x, x11, x21)
        x = self.outc(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.in_channel = in_ch
        self.out_channel = out_ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x, x1):
        # pdb.set_trace()
        x = self.up(x)
        if x.size(1) == self.in_channel:
            pass
        elif (x.size(1) + x1.size(1)) == self.in_channel:
            x = torch.cat([x, x1], dim=1)
        else:
            raise ValueError('Wrong number of channel!')
        x = self.conv(x)
        return x


class dual_up(nn.Module):
    def __init__(self, in_ch, out_ch, conn_list, bilinear=True):
        super(dual_up, self).__init__()
        self.in_channel = in_ch
        self.out_channel = out_ch
        self.conn_list = conn_list
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x, x1, x2):
        x = self.up(x)
        if self.conn_list[0] == 1:
            x = torch.cat([x, x1], dim=1)
        if self.conn_list[1] == 1:
            x = torch.cat([x, x2], dim=1)
        else:
            raise ValueError('Wrong number of channel!')
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class double_conv(nn.Module):
    '''
    (conv => BN => ReLU) * 2
    output_size = input_size
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, output_layer='sigmoid'):
        super(outconv, self).__init__()
        if not output_layer in ['softmax', 'sigmoid', 'tanh']:
            raise ValueError('Wrong category of last layer!')
        if output_layer == 'softmax':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.Softmax(dim=1))
        elif output_layer == 'sigmoid':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.Sigmoid())
        elif output_layer == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.Tanh())

    def forward(self, x):
        x = self.conv(x)
        return x