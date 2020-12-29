"""
MLNet: Multi-Level Net
Input is original image, latent feature and structure feature of segmentation
"""

from .unet_part import *
import pdb


class Unet_1st_stage(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet_1st_stage, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # unet, seg mask
        x1 = self.inc(x)                # 224,224,64
        x2 = self.down1(x1)             # 112,112,128
        x3 = self.down2(x2)             # 56,56,256
        x4 = self.down3(x3)             # 28,28,512

        seg_latent_feat = self.down4(x4)   # 14,14,512

        x = self.up1(seg_latent_feat, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        seg_structure_feat = self.up4(x, x1)      # 224,224,64
        seg_mask = self.outc(seg_structure_feat)

        return seg_mask, seg_structure_feat, seg_latent_feat


class MultiLevelNet(nn.Module):
    def __init__(self, in_ch, modality, ablation_mode=6, image_skip_conn=[0, 0, 0]):
        super(MultiLevelNet, self).__init__()
        self.modality = modality
        self.ablation_mode = ablation_mode
        self.image_skip_conn = image_skip_conn
        # input: 224 * 224 * in_ch
        """
        ablation study mode
        """
        # 0: output_structure                       (1 feature)
        # 1: latent_structure                       (1 feature)
        # 2: image, i.e. auto-encoder               (1 feature)
        # 3: output_structure + latent_structure    (2 features)
        # 4: output_structure + image               (2 features)
        # 5: image + latent_structure    (2 features)
        # 6: full level

        unit_channel = 64
        self.image_encoder = image_encoder(3, unit_channel)

        # input: 224 * 224 * 1
        # 2nd unit channel
        self.inc = inconv(1, unit_channel)
        self.seg_encoder_down1 = down(unit_channel, unit_channel * 2)
        self.seg_encoder_down2 = down(unit_channel * 2, unit_channel * 4)
        self.seg_encoder_down3 = down(unit_channel * 4, unit_channel * 8)
        self.seg_encoder_down4 = down(unit_channel * 8, unit_channel * 8)      # 14 * 14 * 512

        """
        Difference: number channel of decoder
        """
        if ablation_mode == 0:
            # output_structure
            self.up1 = up(1024, 256)
            self.up2 = up(512, 384)
            self.up3 = up(512, 64)
            self.up4 = up(128, 64)
        elif ablation_mode in [1, 2]:
            # 1 feature
            self.up1 = up_wo_skip(512, 512)
            self.up2 = up_wo_skip(512, 256)
            self.up3 = up_wo_skip(256, 128)
            self.up4 = up_wo_skip(128, 64)
        elif ablation_mode in [3, 4]:
            # output_structure + **
            feature_channel = [1536 + self.image_skip_conn[2] * 512,
                               768 + self.image_skip_conn[1] * 256,
                               256 + self.image_skip_conn[0] * 128]
            self.up1 = up(feature_channel[0], 512)
            self.up2 = up(feature_channel[1], 128)
            self.up3 = up(feature_channel[2], 160)
            self.up4 = up_wo_skip(160, 64)
        elif ablation_mode == 5:
            # image + latent_structure
            self.up1 = up_wo_skip(1024, 512)
            self.up2 = up_wo_skip(512, 256)
            self.up3 = up_wo_skip(256, 128)
            self.up4 = up_wo_skip(128, 64)
        else:
            # full, 3 features
            self.up1 = up(2048, 768)
            self.up2 = up(1024, 384)
            self.up3 = up(512, 192)
            self.up4 = up_wo_skip(192, 64)
        self.outc = outconv(64, in_ch)

    def forward(self, image, seg_mask, seg_latent_feat=None):
        # image 2 rec_latent_feat_i
        if self.ablation_mode != '0':
            feat1, feat2, feat3, rec_latent_feat_i = self.image_encoder(image)

        if self.modality == 'oct':
            seg_mask = torch.argmax(seg_mask, dim=1).unsqueeze(dim=1)
            seg_mask = (seg_mask / 11).clamp(0, 1).float()

        # seg_structure_feat 2 rec_latent_feat_s
        x0 = self.inc(seg_mask)
        x1 = self.seg_encoder_down1(x0)     # 112 * 112 * 128
        x2 = self.seg_encoder_down2(x1)                     # 56 * 56 * 256
        x3 = self.seg_encoder_down3(x2)                     # 28 * 28 * 512
        x4 = self.seg_encoder_down4(x3)                     # 14 * 14 * 512

        rec_latent_feat_s = x4
        # 14 * 14 * 1536

        """
        feature fusion with different mode;
        encoder is different
        """
        if self.ablation_mode == 0:
            # only output_structure
            rec_latent_feat = rec_latent_feat_s
            x = self.up1(rec_latent_feat, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x, x0)
        elif self.ablation_mode in [1, 2]:
            # 1 feature without skip
            rec_latent_feat = seg_latent_feat if self.ablation_mode == 1 else rec_latent_feat_i
            x = self.up1(rec_latent_feat)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
        elif self.ablation_mode in [3, 4]:
            # 3: output_structure + latent_structure
            # 4: output_structure + image
            if self.ablation_mode == 3:
                rec_latent_feat = torch.cat([rec_latent_feat_s, seg_latent_feat], dim=1)
            else:
                rec_latent_feat = torch.cat([rec_latent_feat_s, rec_latent_feat_i], dim=1)
            if self.image_skip_conn[2]:
                x3 = torch.cat([x3, feat3], dim=1)
            if self.image_skip_conn[1]:
                x2 = torch.cat([x2, feat2], dim=1)
            if self.image_skip_conn[0]:
                x1 = torch.cat([x1, feat1], dim=1)
            x = self.up1(rec_latent_feat, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x)
        elif self.ablation_mode == 5:
            # output_structure + latent_structure
            rec_latent_feat = torch.cat([rec_latent_feat_i, seg_latent_feat], dim=1)
            x = self.up1(rec_latent_feat)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
        else:
            # full multi-level
            rec_latent_feat = torch.cat([rec_latent_feat_i, rec_latent_feat_s, seg_latent_feat], dim=1)
            x = self.up1(rec_latent_feat, x3)  # 28 * 28 * 768
            x = self.up2(x, x2)  # 56 * 56 * 384
            x = self.up3(x, x1)  # 112 * 112 * 192
            x = self.up4(x)

        x = self.outc(x)

        return x


class image_encoder(nn.Module):
    def __init__(self, in_channel, unit_channel):
        super(image_encoder, self).__init__()
        self.inconv = inconv(in_channel, unit_channel)  # 224 * 224 * 64
        self.down1 = down(unit_channel, unit_channel * 2)
        self.down2 = down(unit_channel * 2, unit_channel * 4)
        self.down3 = down(unit_channel * 4, unit_channel * 8)
        self.down4 = down(unit_channel * 8, unit_channel * 8)      # 14 * 14 * 512

    def forward(self, image):
        x1 = self.inconv(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        return x2, x3, x4, x


def main():
    pass


if __name__ == '__main__':
    main()
