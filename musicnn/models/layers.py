from functools import reduce, partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """Base encoder class specifying the spec
    """
    def __init__(self):
        """"""
        super().__init__()

    def get_hidden_state(self, x, layer):
        """"""
        raise NotImplementedError()
    
    def forward(self, x):
        """"""
        raise NotImplementedError()


class BaseDecoder(nn.Module):
    """Base decoder class to specify the spec

    Args:
        encoder (BaseEncoder): encoder that is to be used for building decoder
    """
    def __init__(self, encoder):
        """"""
        super().__init__()
        self._build_decoder(encoder)
    
    def _build_decoder(self, encoder):
        """"""
        raise NotImplementedError()
    
    def forward(self, z):
        """"""
        raise NotImplementedError()


class VGGlike2DEncoder(BaseEncoder):
    """VGG-like model for encoding 2D music input to latent vector

    Args:
        input_shape (tuple): the size of width and height of input image
        n_hidden (int): number of hidden units of the output of the encoder
        layer1_channels (int): number of filters in first layer. it doubles up
        kernel_size (int): size (both width and height) of the 2D kernel
        pooling (nn.Pooling?): pooling class to be applied to the model
        pool_size (int): scale of the sub-sampling of pooling layer
        non_linearity (nn.(any-non-linearity)): non linearity of the model
        global_average_pooling (bool): indicator for applying GAP.
                                       if false, the hidden activation is flatten
        batch_norm (bool): indicator for applying batch-normalization
    """
    def __init__(self, input_shape=(256, 512), n_hidden=256, n_convs=1,
                 layer1_channels=8, kernel_size=3, pooling=nn.MaxPool2d,
                 pool_size=2, non_linearity=nn.ReLU, global_average_pooling=True,
                 batch_norm=True):
        """"""
        super().__init__()

        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.pooling_layer = pooling
        self.non_linearity = non_linearity
        self.global_average_pooling = global_average_pooling
        self.conv_layers = []
        self.n_convs = n_convs

        # build
        z = test_x = torch.randn(1, 1, input_shape[0], input_shape[1])
        in_chan = 1
        out_chan = layer1_channels
        while z.shape[-1] > 1 and z.shape[-2] > 1:
            # build
            self.conv_layers.append(
                ConvBlock2D(
                    n_convs, in_chan, out_chan, kernel_size,
                    pooling, pool_size, non_linearity, batch_norm
                )
            )
            encoder_ = nn.Sequential(*self.conv_layers)   
            z = encoder_(test_x)

            # update # of input channels
            if out_chan < n_hidden:
                in_chan = out_chan
                out_chan = 2 * in_chan
            else:
                in_chan = out_chan

        # cache this info for building decoder
        # NOTE: not containing the batch dimension!
        self.shape_before_squeeze = z.shape[1:]

        # global average pooling
        # also, cache output hidden shape
        # NOTE: not containing the batch dimension!
        if self.global_average_pooling:
            self.conv_layers.append(GlobalAveragePooling())
            self.shape_hidden = self.n_hidden
        else:  # otherwise, flatten it
            self.conv_layers.append(Flattening())
            self.shape_hidden = self.shape_before_squeeze

        # initiate the encoder
        self.encoder = nn.ModuleList(self.conv_layers)

    def get_hidden_state(self, X, layer=10):
        z = X
        for layer in self.encoder[:layer]:
            z = layer(z)
        return z
    
    def forward(self, X):
        # check the size
        assert X.shape[1:] == (1, self.input_shape[0], self.input_shape[1])
        z = X
        for layer in self.encoder:
            z = layer(z)
        return z


class VGGlike2DDecoder(BaseDecoder):
    """VGG-like model for encoding 2D music input to latent vector

    Args:
        encoder (BaseEncoder): encoder that is to be used for building decoder
    """
    def _build_decoder(self, encoder):
        """"""
        decoder = []
        test_x = torch.randn(1, 1, *encoder.input_shape)
        for i, layer in enumerate(encoder.encoder):
            # inverse of terminal `squeezing` layer
            if isinstance(layer, (GlobalAveragePooling, Flattening)):
                # in this case, we need to span the hidden layer into higher dim
                if isinstance(layer, GlobalAveragePooling):
                    numel = reduce(np.multiply, encoder.shape_before_squeeze)
                    decoder.append(
                        nn.Sequential(
                            nn.Linear(encoder.n_hidden, numel),
                            UnFlattening(encoder.shape_before_squeeze)
                        )
                    )
                else:
                    decoder.append(UnFlattening(encoder.shape_before_squeeze))

            elif isinstance(layer, ConvBlock2D):
                if i == 0:  # terminal layer
                    non_lin = Identity
                else:
                    non_lin = encoder.non_linearity

                target_shape = test_x.shape[2:]  # only spatial dimension
                test_x = layer(test_x)

                decoder.append(TransposedConvBlock2D(
                    layer.n_convs, layer.out_channels, layer.in_channels,
                    layer.kernel_size, layer.pooling, target_shape,
                    non_lin, layer.batch_norm
                ))

        self.decoder = nn.ModuleList(decoder[::-1])

    def forward(self, z):
        """"""
        X = z
        for layer in self.decoder:
            X = layer(X)
        return X


class ConvBlock2D(nn.Module):
    """Convolution block contains conv-pool chain and other layers

    Optionally, can have multiple conv layers and batch_norm / dropout layers
    """
    def __init__(self, n_convs, in_channels, out_channels, kernel_size,
                 pooling, pool_size, non_linearity, batch_norm):
        """"""
        super().__init__()
        self.n_convs = n_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.pool_size = pool_size
        self.batch_norm = batch_norm

        convs = []
        for i in range(n_convs):
            if i == 0:
                in_chan = self.in_channels
            else:
                in_chan = self.out_channels

            convs.append(
                SameConv2D(in_chan, out_channels, kernel_size,
                           batch_norm=batch_norm)
            )
        self.convs = nn.Sequential(*convs)
        self.pool = pooling(pool_size)
        self.non_linearity = non_linearity()
    
    def forward(self, x):
        return self.pool(self.non_linearity(self.convs(x)))


class TransposedConvBlock2D(nn.Module):
    """Transposed ConvBlock2D for building decoder
    """
    def __init__(self, n_convs, in_channels, out_channels, kernel_size,
                 pooling, pool_size, non_linearity, batch_norm):
        """"""
        super().__init__()
        self.n_convs = n_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.pool_size = pool_size
        self.batch_norm = batch_norm

        convs = []
        for i in range(n_convs):
            if i == 0:
                in_chan = self.in_channels
            else:
                in_chan = self.out_channels

            convs.append(
                SameConv2D(in_chan, out_channels, kernel_size,
                           batch_norm=batch_norm)
            )
        self.convs = nn.Sequential(*convs)
        self.unpool = partial(F.interpolate,
                              size=pool_size, mode='bilinear', align_corners=True)
        self.non_linearity = non_linearity()

    def forward(self, x):
        """"""
        return self.non_linearity(self.convs(self.unpool(x)))


class SameConv2D(nn.Module):
    """2-D convolution layer with same sized-output
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, batch_norm=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.batch_norm = batch_norm

        # build padder
        pad = []
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        for size in kernel_size[::-1]:  # width first, height next
            pad.append(size // 2)
            if size % 2 == 0:  # even size
                pad.append(size // 2 - 1)
            else:  # odd size
                pad.append(size // 2)
        pad = tuple(pad)  # convert to tuple
        self.pad = nn.ReflectionPad2d(pad)

        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(self.pad(x))


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling
    """
    def __init__(self):
        super().__init__()
        self.flatten = Flattening()
    
    def forward(self, x):
        return self.flatten(x).mean(dim=-1)


class Flattening(nn.Module):
    """Flatten any spatial dimensions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1)


class UnFlattening(nn.Module):
    """Un-Flatten input based on the original shape
    
    Args:
        original_shape (torch.Size): target output shape. number of element
                                     should be same to the input tensor
    """
    def __init__(self, original_shape):
        super().__init__()
        self.original_shape = original_shape
    
    def forward(self, x):
        # sanity check
        return x.view(x.shape[0], *self.original_shape)


class Identity(nn.Module):
    """Identity module
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    """STFT module
    """
    def __init__(self, n_fft, hop_sz, magnitude=True, log=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_sz = hop_sz
        self.magnitude = magnitude
        self.log = log

    def _magnitude(self, x):
        return (x[..., 0]**2 + x[..., 1]**2)**0.5
    
    def _log(self, x, eps=1e-8):
        eps = torch.tensor([eps]).float()
        if x.is_cuda: eps = eps.cuda()
        return torch.log10(torch.max(x, eps))
    
    def forward(self, x):
        X = torch.stft(x, self.n_fft, self.hop_sz)
        if self.magnitude:
            X = self._magnitude(X)
        if self.log:
            X = self._log(X)
        return X


class StandardScaler(nn.Module):
    """Standard scaler for z-scoring
    """
    def __init__(self, mean, std, eps=1e-10):
        super().__init__()
        self.mean_ = nn.Parameter(torch.FloatTensor(mean))
        self.std_ = nn.Parameter(torch.FloatTensor(np.maximum(std, eps)))

        # to make sure these are not learned
        self.mean_.requires_grad = False
        self.std_.requires_grad = False
    
    def forward(self, x):
        if x.dim() == 2:
            return (x - self.mean_[None]) / self.std_[None]
        elif x.dim() == 3:
            return (x - self.mean_[None, :, None]) / self.std_[None, :, None]
        else:
            raise ValueError('[ERROR] only 2 to 3 dimensional \
                              input is supported!')