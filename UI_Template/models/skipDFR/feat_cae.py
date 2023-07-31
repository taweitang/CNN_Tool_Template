import torch
import torch.nn as nn

#########################################
#    1 x 1 conv CAE
#########################################
class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAE, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
        # of the features for rgb
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]
        # layers += [nn.ReLU()]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss

############################################
# U-Net CAE
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class FeatCAEUNET(nn.Module):
    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAEUNET, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(latent_dim*2, latent_dim, input_nc=None, submodule=None,innermost=True)
        unet_block = UnetSkipConnectionBlock((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(in_channels, (in_channels + 2 * latent_dim) // 2, input_nc=in_channels, submodule=unet_block, outermost=True)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        #if type(norm_layer) == functools.partial:
        #    use_bias = norm_layer.func == nn.InstanceNorm2d
        #else:
        #use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=1,
                             stride=1, padding=0)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=1, stride=1,
                                        padding=0)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=1, stride=1,
                                        padding=0)
            down = [downrelu, downconv]
            up = [upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=1, stride=1,
                                        padding=0)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

########################################################

# #########################################
# #    3 x 3 conv CAE
# #########################################
# class FeatCAE(nn.Module):
#     """Autoencoder."""

#     def __init__(self, in_channels=1000, latent_dim=50):
#         super(FeatCAE, self).__init__()

#         layers = []
#         layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)]

#         self.encoder = nn.Sequential(*layers)

#         # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
#         # of the features for rgb
#         layers = []
#         layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=3, stride=1, padding=1)]
#         layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
#         layers += [nn.ReLU()]
#         layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=3, stride=1, padding=1)]
#         # layers += [nn.ReLU()]

#         self.decoder = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

#     def relative_euclidean_distance(self, a, b):
#         return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

#     def loss_function(self, x, x_hat):
#         loss = torch.mean((x - x_hat) ** 2)
#         return loss

#     def compute_energy(self, x, x_hat):
#         loss = torch.mean((x - x_hat) ** 2, dim=1)
#         return loss

################################################
# Feature AE with Shuffle Group Convolution
################################################

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ChannelShuffle(nn.Module):
    def __init__(groups=1):
        self.groups = groups

    def forward(x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, groups, 
            channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

##########################################
# 3 x 3 group conv with shuffle CAE 
##########################################
class FeatSCAE(nn.Module):
    """Autoencoder with shuffled group convolution."""

    def __init__(self, in_channels=1000, latent_dim=50):
        """
        Note: in_channels and latent_dim has to be even, because we use shuffled group convolution
        """
        super(FeatCAE, self).__init__()
        
        self.groups = [8, 4]
        in_channels2 = (in_channels + 2 * latent_dim) // 2
        in_channels2 = in_channels2 + in_channels2%4
        # Encoder
        # inchannels should be a multiple of the number of groups
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels2, kernel_size=1, stride=1, padding=0, groups=8),
            nn.BatchNorm2d(num_features=in_channels2),
            nn.ReLU(inplace=True))
        self.channel_shuffle1 = ChannelShuffle(groups=8)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels2, 2 * latent_dim, kernel_size=1, stride=1, padding=0, groups=4),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(inplace=True)
        )
        self.channel_shuffle2 = ChannelShuffle(groups=4)

        self.mid_conv = nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.conv3 = nn.Sequential(
            nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2 * latent_dim should be a multiple of the number of groups 4, if latent_dim is a multiple of 2 then it satisfies that condition
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0, groups=4),
            nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2),
            nn.ReLU(inplace=True)
        )
        self.channel_shuffle4 = ChannelShuffle(groups=4)
        
        # (in_channels + 2 * latent_dim) // 2 should be a multiple of the number of groups 8
        self.conv5 = nn.Conv2d(inchannels4, in_channels, kernel_size=1, stride=1, padding=0, groups=8)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.channel_shuffle1(x)
        x = self.conv2(x)
        x = self.channel_shuffle2(x)
        x = self.mid_conv(x)

        # decoder
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.channel_shuffle4(x)
        x = self.conv5(x)
        return x

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=1) / a.norm(2, dim=1)

    def loss_function(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2)
        return loss

    def compute_energy(self, x, x_hat):
        loss = torch.mean((x - x_hat) ** 2, dim=1)
        return loss

    
if __name__ == "__main__":
    import numpy as np
    import time

    device = torch.device("cuda:1")
    x = torch.Tensor(np.random.randn(1, 3000, 64, 64)).to(device)
    feat_ae = FeatCAE(in_channels=3000, latent_dim=200).to(device)

    time_s = time.time()
    for i in range(10):
        time_ss = time.time()
        out = feat_ae(x)
        print("Time cost:", (time.time() - time_ss), "s")

    print("Time cost:", (time.time() - time_s), "s")
    print("Feature (n_samples, n_features):", out.shape)
