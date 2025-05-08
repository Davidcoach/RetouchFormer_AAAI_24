import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import itertools

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings

        min_value, min_encoding_indices = torch.min(d, dim=1)

        min_encoding_indices = min_encoding_indices.unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

# pytorch_diffusion + derived encoder decoder
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm2 = Normalize(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm3 = Normalize(in_channels)
        self.conv3 = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(nonlinearity(self.norm1(x)))
        x = self.conv2(nonlinearity(self.norm2(x)))
        x = F.adaptive_avg_pool2d(x,1)
        x = self.conv3(x)
        return x

class PreQuantConv(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm2 = Normalize(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels,
                                emb_dim,
                                kernel_size=1)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(nonlinearity(self.norm1(x)))
        x = self.conv2(nonlinearity(self.norm2(x)))
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.num = 0

    def forward(self, x, y=None):
        h_ = x
        h_ = self.norm1(h_)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b, self.head_size, self.att_size ,h*w)
        q = q.permute(0, 3, 1, 2).contiguous() # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size ,h*w)
        k = k.permute(0, 3, 1, 2).contiguous()

        v = v.reshape(b, self.head_size, self.att_size ,h*w)
        v = v.permute(0, 3, 1, 2).contiguous()


        q = q.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).transpose(2,3).contiguous()

        scale = int(self.att_size)**(-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous() # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2).contiguous()

        w_ = self.proj_out(w_)

        return x+w_


class MultiHeadEncoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, double_z=True, enable_mid=True,
                 head_size=1, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        hs = {}
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        hs['in'] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions-1:
                # hs.append(h)
                hs['block_'+str(i_level)] = h
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs['block_'+str(i_level)+'_atten'] = h
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            hs['mid_atten'] = h

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # hs.append(h)
        hs['out'] = h

        return hs

class MultiHeadDecoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=16, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class MultiHeadDecoderTransformer(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=16, dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs):
        #assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h, hs['mid_atten'])
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, hs['block_'+str(i_level)+'_atten'])
                    # hfeature = h.clone()
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class VQVAEGAN(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8),
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3,
                 resolution=512, z_channels=256, double_z=False, enable_mid=True,
                 fix_decoder=False, fix_codebook=False, head_size=1, **ignore_kwargs):
        super(VQVAEGAN, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z,
                               enable_mid=enable_mid, head_size=head_size)
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input):
        quant, diff, info, hs = self.encode(input)
        dec = self.decode(quant)

        return dec, diff, info, hs

class VQVAEGANMultiHeadTransformer(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8),
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3,
                 resolution=512, z_channels=256, double_z=False, enable_mid=True,
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, constrastive_learning_loss_weight=0.0,
                 head_size=1):
        super(VQVAEGANMultiHeadTransformer, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z,
                               enable_mid=enable_mid, head_size=head_size)
        self.decoder = MultiHeadDecoderTransformer(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant, hs):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, hs)

        return dec

    def forward(self, input):
        quant, diff, info, hs = self.encode(input)
        dec = self.decode(quant, hs)

        return dec, diff, info, hs

class CycleVQVAEGAN(nn.Module):
    """
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                fix_decoder=False, fix_codebook=False, head_size=1, **ignore_kwargs):
    """
    def __init__(self,fix_decoder=False, fix_codebook=False, **kwargs):
        super(CycleVQVAEGAN, self).__init__()
        self.LQ = self.build_blocks(**kwargs)
        self.HQ = self.build_blocks(**kwargs)

        if fix_decoder:
            for _, param in self.LQ.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.LQ.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.LQ.quantize.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.LQ.quantize.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.quantize.named_parameters():
                param.requires_grad = False

    def build_blocks(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8),
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3,
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, head_size=1, **ignore_kwargs):
        m = nn.Module()
        encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z,
                               enable_mid=enable_mid, head_size=head_size)
        decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)
        quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        m.add_module('encoder',encoder)
        m.add_module('decoder',decoder)
        m.add_module('quantize',quantize)
        m.add_module('quant_conv',quant_conv)
        m.add_module('post_quant_conv',post_quant_conv)
        return m

    def encode_LQ(self, x):
        hs = self.LQ.encoder(x)
        enc_out = hs['out']
        quant_in = self.LQ.quant_conv(hs['out'])
        quant, emb_loss, info = self.LQ.quantize(quant_in)
        return enc_out, quant, emb_loss, info, hs

    def encode_HQ(self, x):
        hs = self.HQ.encoder(x)
        enc_out = hs['out']
        quant_in = self.HQ.quant_conv(hs['out'])
        quant, emb_loss, info = self.HQ.quantize(quant_in)
        return enc_out, quant, emb_loss, info, hs

    def decode_LQ(self, quant):
        quant = self.LQ.post_quant_conv(quant)
        dec = self.LQ.decoder(quant)
        return dec

    def decode_HQ(self, quant):
        quant = self.HQ.post_quant_conv(quant)
        dec = self.HQ.decoder(quant)
        return dec


    def forward(self, input_lq, input_hq=None, stage=None):
        if input_hq is None:
            quant, diff, info, hs = self.encode_LQ(input_lq)
            dec = self.decode_HQ(quant)
            return dec, diff, info, hs

        #assert stage is not None, "Please select stage (HQ or LQ or LQHQ) !"

        if stage == 'HQ':
            # HQ -> LQ -> HQ (a->b->a)
            enc_a, quant_a, diff_a, _, _ = self.encode_HQ(input_hq) # enc HQ
            dec_ab = self.decode_LQ(quant_a) # enc HQ
            enc_ab, quant_ab, diff_ab, _, _  = self.encode_LQ(dec_ab) # enc LQ
            dec_aba = self.decode_HQ(quant_ab) # LQ -> HQ
            # self cycle
            dec_aa = self.decode_HQ(quant_a)
            # enc consistency
            enc_a_with_LQenc = self.LQ.encoder(input_hq)['out']

            enc_diff_a_a = F.mse_loss(enc_a,enc_a_with_LQenc)
            enc_diff_a_ab = F.mse_loss(enc_a,enc_ab)

            return (dec_aa, dec_ab, dec_aba), (diff_a, diff_ab), (enc_diff_a_a, enc_diff_a_ab)
        elif stage== 'LQ': # LQ
            # LQ -> HQ -> LQ (b -> a -> b)
            enc_b, quant_b, diff_b, _, _ = self.encode_LQ(input_lq)
            dec_ba = self.decode_HQ(quant_b)
            enc_ba, quant_ba, diff_ba,_,_ = self.encode_HQ(dec_ba)
            dec_bab = self.decode_LQ(quant_ba)
            # self cycle
            dec_bb = self.decode_LQ(quant_b)
            # enc consistency
            enc_b_with_HQenc = self.HQ.encoder(input_lq)['out']

            enc_diff_b_b = F.mse_loss(enc_b,enc_b_with_HQenc)
            enc_diff_b_ba = F.mse_loss(enc_b,enc_ba)

            return (dec_bb, dec_ba, dec_bab), (diff_b, diff_ba), (enc_diff_b_b, enc_diff_b_ba)
        else:
            #with torch.no_grad():
            # HQ -> LQ -> HQ (a->b->a)
            enc_a, quant_a, diff_a, _, _ = self.encode_HQ(input_hq) # enc HQ
            dec_ab = self.decode_LQ(quant_a) # enc HQ
            enc_ab, quant_ab, diff_ab, _, _  = self.encode_LQ(dec_ab) # enc LQ
            dec_aba = self.decode_HQ(quant_ab) # LQ -> HQ
            # self cycle
            dec_aa = self.decode_HQ(quant_a)
            # enc consistency
            enc_a_with_LQenc = self.LQ.encoder(input_hq)['out']

            enc_diff_a_a = F.mse_loss(enc_a,enc_a_with_LQenc)
            enc_diff_a_ab = F.mse_loss(enc_a,enc_ab)

            # LQ -> HQ -> LQ (b -> a -> b)
            enc_b, quant_b, diff_b, _, _ = self.encode_LQ(input_lq)
            dec_ba = self.decode_HQ(quant_b)
            enc_ba, quant_ba, diff_ba,_,_ = self.encode_HQ(dec_ba)
            dec_bab = self.decode_LQ(quant_ba)
            # self cycle
            dec_bb = self.decode_LQ(quant_b)
            # enc consistency
            enc_b_with_HQenc = self.HQ.encoder(input_lq)['out']

            enc_diff_b_b = F.mse_loss(enc_b,enc_b_with_HQenc)
            enc_diff_b_ba = F.mse_loss(enc_b,enc_ba)
            return (dec_aa,dec_ab,dec_aba,dec_bb,dec_ba,dec_bab), (diff_a,diff_ab,diff_b,diff_ba), (enc_diff_a_a, enc_diff_b_b, enc_diff_a_ab, enc_diff_b_ba)

class DualVQVAEGAN(nn.Module):
    """
    def __init__(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8), 
                num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3, 
                resolution=512, z_channels=256, double_z=False, enable_mid=True, 
                fix_decoder=False, fix_codebook=False, head_size=1, **ignore_kwargs):
    """
    def __init__(self,fix_decoder=False, fix_codebook=False, **kwargs):
        super(DualVQVAEGAN, self).__init__()
        self.encoder = self.build_encoder(**kwargs)
        self.mlp = MLP(2*kwargs['z_channels'] if kwargs['double_z'] else kwargs['z_channels'],)
        self.LQ = self.build_blocks(**kwargs)
        self.HQ = self.build_blocks(**kwargs)

        if fix_decoder:
            for _, param in self.LQ.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.LQ.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.LQ.quantize.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.quantize.named_parameters():
                param.requires_grad = False
        elif fix_codebook:
            for _, param in self.LQ.quantize.named_parameters():
                param.requires_grad = False
            for _, param in self.HQ.quantize.named_parameters():
                param.requires_grad = False

    def build_blocks(self, n_embed=1024, embed_dim=256, ch=128, out_ch=3, ch_mult=(1,2,4,8),
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3,
                 resolution=512, z_channels=256, double_z=False, enable_mid=True, head_size=1, **ignore_kwargs):
        m = nn.Module()
        decoder = MultiHeadDecoderTransformer(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)
        quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        #quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        quant_conv = PreQuantConv(z_channels, embed_dim)
        post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        m.add_module('decoder',decoder)
        m.add_module('quantize',quantize)
        m.add_module('quant_conv',quant_conv)
        m.add_module('post_quant_conv',post_quant_conv)
        return m

    def build_encoder(self,
                      ch=128,
                      out_ch=3,
                      ch_mult=(1, 2, 4, 8),
                      num_res_blocks=2,
                      attn_resolutions=16,
                      dropout=0.0,
                      in_channels=3,
                      resolution=512,
                      z_channels=256,
                      double_z=False,
                      enable_mid=True,
                      head_size=1,
                      **ignore_kwargs):
        encoder = MultiHeadEncoder(ch=ch,
                                        out_ch=out_ch,
                                        ch_mult=ch_mult,
                                        num_res_blocks=num_res_blocks,
                                        attn_resolutions=attn_resolutions,
                                        dropout=dropout,
                                        in_channels=in_channels,
                                        resolution=resolution,
                                        z_channels=z_channels,
                                        double_z=double_z,
                                        enable_mid=enable_mid,
                                        head_size=head_size)
        return encoder

    def encode(self, x):
        hs = self.encoder(x)
        enc_out = hs['out']
        return enc_out, hs

    def encode_z(self, x):
        hs = self.encoder(x)
        z = self.mlp(hs['out'])
        return z

    def decode_LQ(self, x, hs):
        quant = self.LQ.quant_conv(x)
        quant, diff, _ = self.LQ.quantize(quant)
        quant = self.LQ.post_quant_conv(quant)
        dec = self.LQ.decoder(quant, hs)
        return dec, diff

    def decode_HQ(self, x, hs):
        quant = self.HQ.quant_conv(x)
        quant, diff, _ = self.HQ.quantize(quant)
        quant = self.HQ.post_quant_conv(quant)
        dec = self.HQ.decoder(quant, hs)
        return dec, diff

    def forward(self, input_lq, input_hq=None, stage=None):
        if input_hq is None:
            x, _ = self.encode(input_lq)
            dec, diff = self.decode_HQ(x)
            return dec, diff

        assert stage is not None, "Please select stage (self2self or self2other or disc) !"


        if stage == "LQ":
            enc_lq, hs = self.encode(input_lq)
            decLQ_lq, diffLQ_lq = self.decode_LQ(enc_lq, hs)
            decHQ_lq, diffHQ_lq = self.decode_HQ(enc_lq, hs)
            enc_diff_lq = F.triplet_margin_loss(
                self.encode_z(decHQ_lq.detach()),
                self.encode_z(decLQ_lq.detach()),
                self.encode_z(input_hq),
                margin=1.0
                )
            # enc_diff_lq = F.mse_loss(self.encode(decLQ_lq.detach())[0], self.encode(decHQ_lq.detach())[0])
            return (decLQ_lq,decHQ_lq),(diffLQ_lq,diffHQ_lq), enc_diff_lq
        elif stage == "HQ":
            enc_hq, hs = self.encode(input_hq)
            decHQ_hq, diffHQ_hq = self.decode_HQ(enc_hq, hs)
            decLQ_hq, diffLQ_hq = self.decode_LQ(enc_hq, hs)
            enc_diff_hq = F.triplet_margin_loss(
                self.encode_z(decLQ_hq.detach()),
                self.encode_z(decHQ_hq.detach()),
                self.encode_z(input_lq),
                margin=1.0
            )
            #enc_diff_hq = F.mse_loss(self.encode(decHQ_hq.detach())[0], self.encode(decLQ_hq.detach())[0])
            return (decHQ_hq,decLQ_hq),(diffHQ_hq,diffLQ_hq), enc_diff_hq
        else:
            enc_lq, hs_lq = self.encode(input_lq)
            enc_hq, hs_hq = self.encode(input_hq)
            
            decLQ_lq, diffLQ_lq = self.decode_LQ(enc_lq, hs_lq)
            decHQ_lq, diffHQ_lq = self.decode_HQ(enc_lq, hs_lq)

            decHQ_hq, diffHQ_hq = self.decode_HQ(enc_hq, hs_hq)
            decLQ_hq, diffLQ_hq = self.decode_LQ(enc_hq, hs_hq)

            #enc_diff_lq = F.mse_loss(self.encode(decLQ_lq.detach()), self.encode(decHQ_lq.detach()))
            #enc_diff_hq = F.mse_loss(self.encode(decHQ_hq.detach()), self.encode(decLQ_hq.detach()))

            return (decLQ_lq,decHQ_lq,decHQ_hq,decLQ_hq), (diffLQ_lq, diffHQ_lq, diffHQ_hq, diffLQ_hq), (None, None)

class ReluResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.leaky_relu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    
class CSM(nn.Module):
    def __init__(self, num_pred_params):
        super().__init__()
        self.ch = [16, 32, 64, 128]
        self.convs = nn.ModuleList()
        pre_ch = 3
        for ch in self.ch:
            block = nn.Sequential(*[nn.Conv2d(pre_ch,ch,3,padding=1), nn.LeakyReLU()])
            self.convs.append(block)
            pre_ch = ch
        self.convs = nn.Sequential(*self.convs)
        self.final_layer = nn.Sequential(*[nn.Conv2d(self.ch[-1], self.ch[-1], 1),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(self.ch[-1], num_pred_params, 1)
                                         ]) # [b, num_pred_params, res, res]

    def forward(self, x, img):
        sigmoid_params = self.final_layer(self.convs(img))
        alpha, beta = torch.split(sigmoid_params, x.shape[1], dim=1)
        x = torch.sigmoid(x*alpha + beta)
        return x

class MaskGenerator(nn.Module):
    def __init__(self, in_channels, base_ch, ch_mult=(1,2,4,8), num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, base_ch, 3,padding=1)

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = base_ch
        for i in range(1, len(ch_mult)):
            out_ch = base_ch * ch_mult[i]
            block = []
            for j in range(num_res_blocks):
                if j < num_res_blocks-1:
                    block.append(ReluResnetBlock(in_ch,dropout=dropout))
                else:
                    block.append(ReluResnetBlock(in_ch,out_ch,dropout=dropout))
            self.encoder.append(nn.Sequential(*block))
            self.downs.append(Downsample(out_ch, with_conv=True))
            in_ch = out_ch


        self.bottom_block = nn.Sequential(*[ReluResnetBlock(base_ch*ch_mult[-1], base_ch*ch_mult[-1], dropout=dropout), 
                                            Upsample(in_ch, with_conv=True)
                                            ])
        self.skips = nn.ModuleList()
        for i in range(len(ch_mult)):
            self.skips.insert(0, nn.Conv2d(base_ch*ch_mult[i],base_ch*ch_mult[i],kernel_size=1))

        self.decoder = nn.ModuleList()
        self.ups =  nn.ModuleList()
        self.toMasks = nn.ModuleList()
        #self.csms = nn.ModuleList()
        in_ch = base_ch*ch_mult[-1]
        for i in range(len(ch_mult)-2, -1,-1):
            out_ch = base_ch * ch_mult[i]
            block  = []
            for j in range(num_res_blocks):
                if j==0:
                    block.append(ReluResnetBlock(in_ch*2,in_ch,dropout=dropout))
                elif j < num_res_blocks-1:
                    block.append(ReluResnetBlock(in_ch,dropout=dropout))
                else:
                    block.append(ReluResnetBlock(in_ch,out_ch,dropout=dropout))
            self.decoder.append(nn.Sequential(*block))
            #self.csms.append(CSM(2))
            if i>0:
                self.toMasks.append(nn.Conv2d(out_ch, 1, 1))
                self.ups.append(Upsample(out_ch,with_conv=True))
            in_ch=out_ch
        #self.norm_out = Normalize(out_ch)
        self.toHead1 = ReluResnetBlock(out_ch,out_ch)
        self.toHead2 = ReluResnetBlock(out_ch,out_ch)
        self.to_mask1 = nn.Conv2d(out_ch,1,kernel_size=1)
        self.to_mask2 = nn.Conv2d(out_ch,1,kernel_size=1)

        self.adapter = nn.ModuleList()
        self.adapter.append(self.build_Adapter(128,128,512,64))
        self.adapter.append(self.build_Adapter(128,128,512,64))
        self.adapter.append(self.build_Adapter(64,256,512,64))
        self.adapter.append(self.build_Adapter(64,256,512,64))
        self.adapter.append(self.build_Adapter(32,512,512,64))
        self.adapter.append(self.build_Adapter(32,512,512,64))


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='leaky_relu')

        #self.toMask = nn.Conv2d(out_ch, 1, 1)

    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv2d') != -1:
    #         torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='leaky_relu')
    #         #torch.init.xavier_normal_(m.weight.data)
    #         torch.nn.init.constant_(m.bias.data, 0.0)

    def build_Adapter(self, s_ch, s_res, t_ch, t_res):
        s_res_log = int(math.log2(s_res))
        t_res_log = int(math.log2(t_res))

        m = []
        
        for i in range(s_res_log, t_res_log, -1):
            m.append(Downsample(s_ch,with_conv=True))
            m.append(nn.Conv2d(s_ch,s_ch*2,kernel_size=3,padding=1))
            m.append(nn.LeakyReLU())
            s_ch = 2*s_ch
        
        m.append(nn.Conv2d(s_ch, t_ch, kernel_size=1, padding=0))
        m = nn.Sequential(*m)
        return m    

    
    def forward(self, img, diff):
        x = torch.cat([img,diff],dim=1)
        hs = []
        dec_feats = []
        
        x = self.conv0(x)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            hs.insert(0,x)
            x = self.downs[i](x)

        x = self.bottom_block(x)

        masks = []
        for i in range(len(self.decoder)):
            y = self.skips[i](hs[i])
            x = torch.cat([x,y],dim=1)
            x = self.decoder[i](x)
            dec_feats.append(x)
            if i < len(self.toMasks):
                mask = self.toMasks[i](x)
                #mask = self.csms[i](mask,F.interpolate(img,size=x.shape[-2:],mode='bilinear',align_corners=True))
                masks.append(mask)
            if i < len(self.ups):
                x = self.ups[i](x)
        
        head2_feat = self.toHead2(x)
        head2 = torch.sigmoid(self.to_mask2(head2_feat))

        head1_feat =self.toHead1(x)
        # head1 = self.to_mask1(torch.cat([head1_feat,head2_feat],dim=1))
        head1 = self.to_mask1(head1_feat+head2_feat)
        masks.append(head1)

        dec_feats = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in dec_feats))
        for i in range(len(dec_feats)):
            dec_feats[i] = self.adapter[i](dec_feats[i])
        dec_feats = torch.stack(dec_feats, dim=2)

        #mask = self.toMask(x)
        #alpha, beta = self.AdaParams(x).split(1, dim=1)
        #mask = torch.sigmoid(alpha*mask + beta)
        return masks, head2, dec_feats

class VQVAEMaskGAN(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, ch=64, out_ch=3, ch_mult=(1,2,2,4,4,8),
                 num_res_blocks=2, attn_resolutions=[16], dropout=0.0, in_channels=3,
                 resolution=512, z_channels=256, double_z=False, enable_mid=True,
                 fix_decoder=True, fix_codebook=True, fix_encoder=True, head_size=1, **ignore_kwargs):
        super(VQVAEMaskGAN, self).__init__()

        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, double_z=double_z,
                               enable_mid=enable_mid, head_size=head_size)
        self.decoder = MultiHeadDecoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, enable_mid=enable_mid, head_size=head_size)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

        self.mask_generator = MaskGenerator(6, base_ch=32, dropout=dropout)


        if fix_encoder:
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False
        if fix_decoder:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for _, param in self.post_quant_conv.named_parameters():
                param.requires_grad = False
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False
            for _, param in self.quant_conv.named_parameters():
                param.requires_grad = False                
        if fix_codebook:
            for _, param in self.quantize.named_parameters():
                param.requires_grad = False

    def encode(self, x):

        hs = self.encoder(x)
        h = self.quant_conv(hs['out'])
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        return dec

    def forward(self, input, stage="dict"):
        assert stage in ['dict','mask','mix']
        if stage == 'dict':
            quant, diff, info, hs = self.encode(input)
            dec = self.decode(quant)
            return dec, diff, info, hs
        elif stage == 'mix':
            quant, _, _, _ = self.encode(input)
            dec = self.decode(quant)
            head1, head2, ext_feats = self.mask_generator(input, dec)
            return head1, head2, ext_feats
        else:
            lq_img, hq_img = input
            head1, head2, ext_feats = self.mask_generator(lq_img, hq_img)
            return head1, head2, ext_feats
            
