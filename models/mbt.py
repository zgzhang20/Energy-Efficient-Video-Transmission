# class mbt(MeanScaleHyperprior):
r"""Joint Autoregressive Hierarchical Priors model from D.
Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

.. code-block:: none

              ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
        x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
              └───┘    │     └───┘     └───┘        EB        └───┘ │
                       ▼                                            │
                     ┌─┴─┐                                          │
                     │ Q │                                   params ▼
                     └─┬─┘                                          │
                 y_hat ▼                  ┌─────┐                   │
                       ├──────────►───────┤  CP ├────────►──────────┤
                       │                  └─────┘                   │
                       ▼                                            ▼
                       │                                            │
                       ·                  ┌─────┐                   │
                    GC : ◄────────◄───────┤  EP ├────────◄──────────┘
                       ·     scales_hat   └─────┘
                       │      means_hat
                 y_hat ▼
                       │
              ┌───┐    │
    x_hat ──◄─┤g_s├────┘
              └───┘

    EB = Entropy bottleneck
    GC = Gaussian conditional
    EP = Entropy parameters network
    CP = Context prediction (masked convolution)

Args:
    N (int): Number of channels
    M (int): Number of channels in the expansion layers (last layer of the
        encoder and last layer of the hyperprior decoder)
"""
import torch
import torch.nn as nn
from models.pytorch_gdn import GDN

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class mbt_ga(nn.Module):
    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, 24, kernel_size=5, stride=2),
        )

    def forward(self, x):
        y = self.g_a(x)

        return y


class mbt_gs(nn.Module):

    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_s = nn.Sequential(
            deconv(24, N, kernel_size=5, stride=2),
            GDN(N,device, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N,device, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N,device, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x):
        y = self.g_s(x)

        return y


class wz_ga(nn.Module):

    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, 24, kernel_size=5, stride=2),
        )

    def forward(self, x):
        y = self.g_a(x)

        return y


class wz_gs(nn.Module):

    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_s = nn.Sequential(
            deconv(3*24, N, kernel_size=5, stride=2),
            GDN(N,device, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N,device, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N,device, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x):
        y = self.g_s(x)

        return y

class si_ga(nn.Module):

    def __init__(self, N=192, M=192):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N,device),
            conv(N, 24, kernel_size=5, stride=2),
        )

    def forward(self, x):
        y = self.g_a(x)

        return y

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class FasterNet(nn.Module):
    def __init__(self, embed_dim):
        super(FasterNet, self).__init__()
        self.PatchEmbed = nn.Sequential(nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1),
                                        GDN(embed_dim, device),
                                        nn.PReLU()
                                        )
        self.PatchMerging = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                                          GDN(embed_dim, device),
                                          nn.PReLU()
                                          )
        self.Partial_conv3 = nn.Sequential(Partial_conv3(embed_dim, 2),
                                           nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
                                           GDN(embed_dim, device),
                                           nn.PReLU(),
                                           nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
                                           )
        self.Conv = nn.Sequential(nn.Conv2d(embed_dim, 24, kernel_size=3, stride=2, padding=1),
                      GDN(24, device),
                      nn.PReLU()
                      )

    def forward(self, x):
        x = self.PatchEmbed(x)
        x = self.PatchMerging(x)
        x = x + self.Partial_conv3(x)
        x = self.PatchMerging(x)
        x = x + self.Partial_conv3(x)
        x = self.Conv(x)
        return x



