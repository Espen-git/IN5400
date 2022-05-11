import torch.nn as nn
import torchvision
import torch

# Additional/supporting methods and code for the 2021 IN5400 lab on image segmentation.
# Some of this code has been copied or heavily inspired by chunks found online.

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn2(self.relu2(self.conv2(self.bn1(self.relu1(self.conv1(x))))))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs_cropped = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs_cropped], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, chs=[3, 64, 128, 256, 512, 1024], num_class=1):
        super().__init__()
        self.encoder = Encoder(chs)
        self.decoder = Decoder(chs[:0:-1])
        self.head = nn.Conv2d(chs[1], num_class, 1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        encoder_features_inv_order = encoder_features[::-1]
        out = self.decoder(encoder_features_inv_order[0], encoder_features_inv_order[1:])
        out = self.head(out)
        return out
