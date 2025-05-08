from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from model.network_vrt_pair_qkv import Stage
from model.modules.soft_mask_generation import VQVAEMaskGAN
from model.gpen_model import Decoder, Encoder

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0,
                 scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.scale_factor = scale_factor 
    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=False)

class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        # encoder
        self.encoder = Encoder(
            size = 512,
            channel_multiplier=2,
            narrow=1,
            device='cuda')
        
        # decoder
        self.decoder = Decoder(
            size = 512,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            isconcat=True,
            narrow=1,
            device='cuda'
        )
        
        self.vrt = Stage(in_dim = 512,
                 dim = 512,
                 input_resolution = (6, 64, 64),
                 depth = 7,
                 num_heads = 8,
                 window_size = [6, 16, 16],
                 mul_attn_ratio=0.6,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False)
        self.norm = nn.LayerNorm(512)
        
        self.soft_mask = VQVAEMaskGAN()
        self.conv_512 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.conv_256 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.conv_128 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1), nn.LeakyReLU()
        )
        self.back_512 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            deconv(128, 64, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
        )
        self.back_256 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU(),
            deconv(256, 128, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU()
        )
        self.back_128 = nn.Sequential(
            deconv(512, 256, kernel_size=3, padding=1, scale_factor=2), nn.LeakyReLU()
        )
        self.back_65 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.back_64 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )
        self.back_63 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.LeakyReLU()
        )

    def forward(self, source_tensor):
        decoder_noise = self.encoder(source_tensor)
        vrt = torch.stack([self.conv_512(decoder_noise[0]), 
                           self.conv_256(decoder_noise[1]), 
                           self.conv_128(decoder_noise[2]), 
                           decoder_noise[3], decoder_noise[4], decoder_noise[5]], dim=2)
        B, C, T, H, W = vrt.shape
        _, _, attention_feat = self.attention_feat(source_tensor, stage='mix')
        attention_feat = torch.sigmoid(attention_feat)
        vrt_out = self.vrt(vrt, attention_feat.reshape(B, C, T, H, W)).reshape(B, T, H, W, C)
        vrt_out = self.norm(vrt_out).reshape(T, B, C, H, W)
        attention_feat = attention_feat.reshape(T, B, C, H, W)
        vrt = vrt.reshape(T, B, C, H, W)
        attention_list = []
        for atten in attention_feat:
            attention_list.append(atten)
        feature = [512, 256, 128, 65, 64, 63]
        for i in range(6):
            mask = attention_feat[i] > 0.5
            M = torch.zeros_like(attention_feat[i])
            M[mask] = vrt_out[i][mask]
            M[~mask] = vrt[i][~mask]
            block = getattr(self, f"back_{feature[i]}")
            decoder_noise[i] = decoder_noise[i] + block(M)
        result = self.decoder(decoder_noise[::-1])
        return result, attention_list

# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32
        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(5, 5),
                          stride=(2, 2),
                          padding=(1, 1),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 1,
                          nf * 2,
                          kernel_size=(5, 5),
                          stride=(2, 2),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 2,
                          nf * 4,
                          kernel_size=(5, 5),
                          stride=(2, 2),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 4,
                          nf * 8,
                          kernel_size=(5, 5),
                          stride=(1, 1),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(nf * 8,
                          nf * 16,
                          kernel_size=(5, 5),
                          stride=(1, 1),
                          padding=(2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 16,
                      nf * 16,
                      kernel_size=(5, 5),
                      stride=(1, 1),
                      padding=(2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        feat = self.conv(xs)
        return feat


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class img_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x, img):
        sigmoid_params = self.convs(img) # torch.Size([1, 2, res, res])
        alpha, beta = torch.split(sigmoid_params, 512, dim=1)
        return torch.sigmoid(x*alpha + beta)
