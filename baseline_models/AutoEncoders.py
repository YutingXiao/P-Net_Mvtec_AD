from .encder_decoder_part import *


class BaseAutoEncoder(nn.Module):
    def __init__(self, n_channels=1, out_channels=1):
        super(BaseAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            inconv(n_channels, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 1024)
        )
        self.decoder = nn.Sequential(
            up_wo_skip(1024, 512),
            up_wo_skip(512, 256),
            up_wo_skip(256, 128),
            up_wo_skip(128, 64),
            outconv(64, out_channels)
        )

    def forward(self, image):
        x = self.encoder(image)
        x = self.decoder(x)
        return x


class GANomaly(BaseAutoEncoder):
    def __init__(self):
        super(GANomaly, self).__init__()

    def forward(self, image):
        latent_0 = self.encoder(image)
        image_rec = self.decoder(latent_0)
        latent_1 = self.encoder(image_rec)
        return image_rec, latent_0, latent_1

