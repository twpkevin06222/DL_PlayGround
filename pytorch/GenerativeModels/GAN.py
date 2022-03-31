import numpy as np
import torch
from torch import Tensor
from torchsummary import summary
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

seed = 5243

torch.manual_seed(seed)

# %%
# hyperparams
data_path = "../../../data/celeba"
batch_size = 128
img_size = 64
# number of channels
nc = 3
# latent space dimension
latent = 100
# generator feature maps
fm_g = 32
# discriminator feature maps
fm_d = 32
epochs = 10
lr = 0.0002
beta1 = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

# define the transformation needed
trans = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = dset.ImageFolder(root=data_path,
                           transform=trans)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

# plot grid images
# img_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Train Images")
# plt.imshow(np.transpose(vutils.make_grid(img_batch[0].to(device)[:64], padding=1,
#                                          normalize=True).cpu(), (1,2,0)))
# plt.savefig('../fig/example_img.png', dpi=300, format='png')
# plt.show()


# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple = (3, 3)) -> None:
        super(UpBlock, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, activation='relu') -> Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        if activation is 'relu':
            x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim: int, init_filter: int, depth: int) -> None:
        super(Generator, self).__init__()
        self.first = UpBlock(latent_dim, init_filter*2**(depth+1))
        self.upsample = nn.ModuleList([UpBlock(init_filter*(2**(i+1)), init_filter*(2**i))
                                       for i in range(depth, 0, -1)])
        self.last = UpBlock(init_filter*2, 3)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.first(x)
        for up in self.upsample:
            x = up(x)
        x = self.last(x, activation=None)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, init_filter):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, init_filter, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(init_filter, init_filter * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(init_filter * 2, init_filter * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(init_filter * 4, init_filter * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(init_filter * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


netG = Generator(latent, fm_g, 4).to(device)
netD = Discriminator(fm_d).to(device)
netG.apply(weights_init)
netD.apply(weights_init)
# summary(netG, (latent, 1, 1))
# summary(netD, (3, 64, 64))
