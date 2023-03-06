import torch
import torch.nn as nn

class LightweightEncoderDecoder(nn.Module):
    def __init__(self):
        super(LightweightEncoderDecoder, self).__init__()
        
        self.mask_conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, mask):
        #mask = self.mask_conv(mask)
        x = x * mask
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class InpaintingHead(nn.Sequential):
    def __init__(self):
        deconv = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        act = nn.Tanh()
        super().__init__(deconv, act)

class InpaintingHead2(nn.Sequential):
    def __init__(self):
        deconv1 = nn.ConvTranspose2d(256, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        deconv2 = nn.ConvTranspose2d(3, 3, kernel_size=6, stride=4, padding=1, output_padding=0)
        act = nn.Tanh()
        super().__init__(deconv1, deconv2, act)

class XModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        model.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(inplace=True)
        )
        self.model = model
        self.final = nn.Sequential(
            nn.ConvTranspose2d(6, 3, kernel_size=1, stride=1, padding=0, output_padding=0),
            nn.Tanh())
        
    def forward(self, x):
        y = self.model(x)
        y = torch.cat((y, x[:, 0:3, :, :]), 1)
        y = self.final(y)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 6, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32 * 2, 6, 3, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2, 32 * 4, 6, 3, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 4, 32 * 8, 6, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 8, 1, 6, 3, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def initialise_model(model, device):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            #nn.init.constant_(m.bias, 0)
    model.to(device)