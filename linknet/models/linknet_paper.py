from __future__ import division

import functools
import numpy as np
import sys

import chainer
import chainer.functions as F
import chainer.links as L
# from chainer import chain

from chainercv.transforms import resize
from chainercv.utils import download_model
from linknet.models.spatial_dropout import spatial_dropout

from chainer import Variable
from chainercv.links import PixelwiseSoftmaxClassifier


def parse_dict(dic, key, value=None):
    return value if not key in dic else dic[key]

def _without_cudnn(f, x):
    with chainer.using_config('use_cudnn', 'never'):
        return f.apply((x,))[0]


class Conv(chainer.Chain):
    "Convolution2D for inference module"
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False, outsize=None):
        super(Conv, self).__init__()
        with self.init_scope():
            if upsample:
                self.conv = L.Deconvolution2D(
                                in_ch, out_ch, ksize, stride, pad,
                                nobias=nobias, outsize=outsize)
            else:
                if dilation > 1:
                    self.conv = L.DilatedConvolution2D(
                        in_ch, out_ch, ksize, stride, pad, dilation, nobias=nobias)
                else:
                    self.conv = L.Convolution2D(
                        in_ch, out_ch, ksize, stride, pad, nobias=nobias)

    def __call__(self, x):
        return self.conv(x)

    def predict(self, x):
        return self.conv(x)


class ConvBN(Conv):
    """Convolution2D + Batch Normalization"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False, outsize=None):
        super(ConvBN, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                     dilation, nobias, upsample, outsize)

        self.add_link("bn", L.BatchNormalization(out_ch, eps=1e-5, decay=0.95))

    def __call__(self, x):
        return self.bn(self.conv(x))

    def predict(self, x):
        return self.bn(self.conv(x))

class ConvReLU(Conv):
    """Convolution2D + ReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False, outsize=None):
        super(ConvReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                        dilation, nobias, upsample, outsize)

    def __call__(self, x):
        return F.relu(self.conv(x))

    def predict(self, x):
        return F.relu(self.conv(x))


class ConvBNReLU(ConvBN):
    """Convolution2D + Batch Normalization + ReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False, outsize=None):
        super(ConvBNReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                          dilation, nobias, upsample, outsize)


    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))

    def predict(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvPReLU(Conv):
    """Convolution2D + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False, outsize=None):
        super(ConvPReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                        dilation, nobias, upsample, outsize)

        self.add_link("prelu", L.PReLU())

    def __call__(self, x):
        return self.prelu(self.conv(x))

    def predict(self, x):
        return self.prelu(self.conv(x))


class ConvBNPReLU(ConvBN):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=False, outsize=None):
        super(ConvBNPReLU, self).__init__(in_ch, out_ch, ksize, stride, pad,
                                          dilation, nobias, upsample, outsize)

        self.add_link("prelu", L.PReLU())

    def __call__(self, x):
        return self.prelu(self.bn(self.conv(x)))

    def predict(self, x):
        return self.prelu(self.bn(self.conv(x)))


class SymmetricConvPReLU(chainer.Chain):
    """Convolution2D + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=None):
        super(SymmetricConvPReLU, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_ch, out_ch, (ksize, 1), stride, pad, nobias=nobias)
            self.conv2 = L.Convolution2D(
                in_ch, out_ch, (1, ksize), stride, pad, nobias=nobias)
            self.prelu = L.PReLU()

    def __call__(self, x):
        return self.prelu(self.conv2(self.conv1(x)))

    def predict(self, x):
        return self.prelu(self.conv2(self.conv1(x)))


class SymmetricConvBNPReLU(SymmetricConvPReLU):
    """Convolution2D + Batch Normalization + PReLU"""
    def __init__(self, in_ch, out_ch, ksize, stride=1, pad=1, dilation=1,
                 nobias=False, upsample=None):
        super(SymmetricConvBNPReLU, self).__init__(in_ch, out_ch, ksize,
                                                   stride, pad, dilation,
                                                   nobias, upsample)
        self.add_link("bn", L.BatchNormalization(out_ch, eps=1e-5, decay=0.95))

    def __call__(self, x):
        h = self.conv2(self.conv1(x))
        return self.prelu(self.bn(h))

    def predict(self, x):
        h = self.conv2(self.conv1(x))
        return self.prelu(self.bn(h))


class InitialBlock(chainer.Chain):
    """Initial Block"""
    def __init__(self, in_ch=3, out_ch=13, ksize=3, stride=2, pad=1,
                 nobias=False, psize=3, use_bn=True, use_prelu=False):
        super(InitialBlock, self).__init__()
        with self.init_scope():
            this_mod = sys.modules[__name__]
            conv_type = "ConvBN" if use_bn else "Conv"
            activation = "PReLU" if use_prelu else "ReLU"
            conv_type = conv_type + activation
            ConvBlock = getattr(this_mod, conv_type)
            self.conv = ConvBlock(in_ch, out_ch, ksize, stride,
                                  pad, nobias=nobias)
        self.psize = psize
        self.ppad = int((psize - 1) / 2)

    def __call__(self, x):
        x = self.conv(x)
        return F.max_pooling_2d(x, self.psize, 2, self.ppad)

    def predict(self, x):
        x = self.conv(x)
        return F.max_pooling_2d(x, self.psize, 2, self.ppad)


class ResBacisBlock(chainer.Chain):
    """Basic block of ResNet18 and ResNet34"""
    def __init__(self, in_ch=3, out_ch=13, downsample=False, use_bn=True):
        super(ResBacisBlock, self).__init__()
        self.downsample = downsample
        with self.init_scope():
            this_mod = sys.modules[__name__]
            conv_type = "ConvBN" if use_bn else "Conv"
            ConvBlock = getattr(this_mod, conv_type + "ReLU")
            self.conv1 = ConvBlock(in_ch, out_ch, 3, 1, 1, nobias=True)
            ConvBlock = getattr(this_mod, conv_type)
            self.conv2 = ConvBlock(in_ch, out_ch, 3, 1, 1, nobias=True)
            if self.downsample:
                self.conv3 = ConvBlock(in_ch, out_ch, 1, 2, 0, nobias=True)

    def __call__(self, x):
        h1 = self.covn2(self.conv(x))
        if self.downsample:
            return F.relu(h1 + self.conv3(x))
        else:
            return F.relu(h1 + x)


class ResBlock18(chainer.Chain):
    """Initial Block"""
    def __init__(self, use_bn=True, train=True):
        super(ResBlock18, self).__init__()
        self.train = train
        with self.init_scope():
            self.block1_1 = ResBacisBlock(64, 64, use_bn=use_bn)
            self.block1_2 = ResBacisBlock(64, 64, use_bn=use_bn)
            self.block2_1 = ResBacisBlock(64, 128, use_bn=use_bn,
                                          downsample=True)
            self.block2_2 = ResBacisBlock(128, 128, use_bn=use_bn)
            self.block3_1 = ResBacisBlock(128, 256, use_bn=use_bn,
                                          downsample=True)
            self.block3_2 = ResBacisBlock(256, 256, use_bn=use_bn)
            self.block4_1 = ResBacisBlock(256, 512, use_bn=use_bn,
                                          downsample=True)
            self.block4_2 = ResBacisBlock(512, 512, use_bn=use_bn)

    def pytorch2chainer(path):
        pass

    def __call__(self, x):
        with chainer.using_config('train', self.train):
            h1 = self.block1_2(self.block1_1(x))
            h2 = self.block2_2(self.block2_1(h1))
            h3 = self.block3_2(self.block3_1(h2))
            h4 = self.block4_2(self.block4_1(h3))
            return h1, h2, h3, h4

    def predict(self, x):
        x = self.block1_2(self.block1_1(x))
        x = self.block2_2(self.block2_1(x))
        x = self.block3_2(self.block3_1(x))
        return self.block4_2(self.block4_1(x))


class DecoderBlock(chainer.Chain):
    """DecoderBlock Abstract"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, ksize=3, stride=1, pad=1,
                 residual=False, nobias=False, outsize=None,
                 upsample=False, p=None, use_bn=True, use_prelu=False):
        super(DecoderBlock, self).__init__()
        self.residual = residual
        with self.init_scope():
            this_mod = sys.modules[__name__]
            conv_type = "ConvBN" if use_bn else "Conv"
            activation = "PReLU" if use_prelu else "ReLU"
            ConvBlock = getattr(this_mod, conv_type + activation)
            self.conv1 = ConvBlock(in_ch, mid_ch, 1, 1, 0, nobias=True)

            conv_type2 = conv_type + activation
            ConvBlock = getattr(this_mod, conv_type2)
            self.conv2 = ConvBlock(mid_ch, mid_ch, ksize, stride, pad,
                                   nobias=False,
                                   upsample=upsample,
                                   outsize=outsize)

            ConvBlock = getattr(this_mod, conv_type)
            self.conv3 = ConvBlock(mid_ch, out_ch, 1, 1, 0, nobias=True)

    def __call__(self, x):
        h1 = self.conv1(x)
        h1 = self.conv2(h1)
        h1 = self.conv3(h1)
        if self.residual:
            return F.relu(h1 + x)
        return F.relu(h1)

    def predict(self, x):
        h1 = self.conv1(x)
        h1 = self.conv2(h1)
        h1 = self.conv3(h1)
        if self.residual:
            return F.relu(h1 + x)
        return F.relu(h1)


class FullConv(chainer.Chain):
    """FullConv Abstract"""
    def __init__(self, in_ch=3, mid_ch=0, out_ch=13, ksize=3, stride=1, pad=1):
        super(FullConv, self).__init__()
        with self.init_scope():
            self.deconv = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad)

    def __call__(self, x):
        return self.deconv(x)

    def predict(self, x):
        return self.deconv(x)


class LinkNetBasic(chainer.Chain):
    """LinkNet Basic for semantic segmentation."""
    def __init__(self, config, pretrained_model=None):
        super(LinkNetBasic, self).__init__()
        n_class = None
        this_mod = sys.modules[__name__]
        pretrained_path = parse_dict(pretrained_model, 'path', None)
        size = parse_dict(config, 'size', (512, 1024))
        with self.init_scope():
            BlockType = getattr(this_mod, config['initial_block']['type'])
            self.initial_block = BlockType(**config['initial_block']['args'])

            BlockType = getattr(this_mod, config['resblock']['type'])
            self.resblock = BlockType(**config['resblock']['args'])

            BlockType = getattr(this_mod, config['decoder_block4']['type'])
            args = self.parse_outsize(config, 'decoder_block4')
            self.decoder4 = BlockType(**args)

            BlockType = getattr(this_mod, config['decoder_block3']['type'])
            args = self.parse_outsize(config, 'decoder_block3')
            self.decoder3 = BlockType(**args)

            BlockType = getattr(this_mod, config['decoder_block2']['type'])
            args = self.parse_outsize(config, 'decoder_block2')
            self.decoder2 = BlockType(**args)

            BlockType = getattr(this_mod, config['decoder_block1']['type'])
            args = self.parse_outsize(config, 'decoder_block1')
            self.decoder1 = BlockType(**args)

            BlockType = getattr(this_mod, config['finalblock1']['type'])
            self.finalblock1 = BlockType(**config['finalblock1']['args'])

            BlockType = getattr(this_mod, config['finalblock2']['type'])
            self.finalblock2 = BlockType(**config['finalblock2']['args'])

            BlockType = getattr(this_mod, config['finalblock3']['type'])
            self.finalblock3 = BlockType(**config['finalblock3']['args'])

        if pretrained_path:
            chainer.serializers.load_npz(pretrained_path, self)

    def parse_outsize(self, config, key):
        args = config[key]['args']
        scale = config[key]['scale']
        args['outsize'] = (size[0]/scale, size[1]/scale)
        return args


    def __call__(self, x):
        x = self.initial_block(x)
        h1, h2, h3, h4 = self.resblock(x)
        x = self.decoder4(h4)
        x += h3
        x = self.decoder3(x)
        x += h2
        x = self.decoder2(x)
        x += h1
        x = self.decoder1(x)

        x = self.finalblock1(x)
        x = self.finalblock2(x)
        x = self.finalblock3(x)
        return x

    def predict(self, x):
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            x = self.xp.asarray(x)
            if x.ndim == 3:
                x = self.xp.expand_dims(x, 0)
            x = self.initial_block(x)
            h1, h2, h3, h4 = self.resblock(x)
            x = self.decoder4(h4)
            x += h3
            x = self.decoder3(x)
            x += h2
            x = self.decoder2(x)
            x += h1
            x = self.decoder1(x)

            x = self.finalblock1(x)
            x = self.finalblock2(x)
            x = self.finalblock3(x)
            label = self.xp.argmax(x.data, axis=1).astype("i")
            label = chainer.cuda.to_cpu(label)
            return list(label)
