import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .resnet_model import *
from .aspp import build_aspp

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class CGSANet(nn.Module):
    # n_channels: input image channels
    def __init__(self, n_channels = 3):
        super(CGSANet,self).__init__()
        # self.shape = shape
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(pretrained=True)
        ## -------------Encoder--------------
        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        #stage 1
        self.encoder1 = resnet.layer1 #256^2*64
        #stage 2
        self.encoder2 = resnet.layer2 #128^2*128
        #stage 3
        self.encoder3 = resnet.layer3 #64^2*256
        #stage 4
        self.encoder4 = resnet.layer4 #32^2*512

        self.aspp = build_aspp('resnet', 16, nn.BatchNorm2d)

        ## ------------- edge -------------
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = BasicBlock(21,21)
        self.conv1_3_down = nn.Conv2d(21,1,3, padding=1)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = BasicBlock(21,21)
        self.conv2_3_down = nn.Conv2d(21,1,3, padding=1)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = BasicBlock(21,21)
        self.conv3_3_down = nn.Conv2d(21, 1, 3, padding=1)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = BasicBlock(21,21)
        self.conv4_3_down = nn.Conv2d(21, 1, 3, padding=1)

        ## edge bilinear upsampling
        self.upscore2e = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore3e = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore4e = nn.Upsample(scale_factor=8, mode='bilinear')

        self.edgev = nn.Conv2d(4, 1, 1)
        self.edgef = nn.Conv2d(1, 1, 1)

        ## -------------Decoder--------------
        self.conv_bn_relu5d_1 = ConvBnRelu(256+512,512, 3, 1,3 // 2)
        self.conv_bn_relu5d_m = ConvBnRelu(512,512, 3, 1,3 // 2)
        self.conv_bn_relu5d_2 = ConvBnRelu(512,512, 3, 1,3 // 2)

        self.conv_bn_relu4d_1 = ConvBnRelu(256+512,512, 3, 1,3 // 2)
        self.conv_bn_relu4d_m = ConvBnRelu(512,512, 3, 1,3 // 2)
        self.conv_bn_relu4d_2 = ConvBnRelu(512,512, 3, 1,3 // 2)

        self.conv_bn_relu3d_1 = ConvBnRelu(256+512,512, 3, 1,3 // 2)
        self.conv_bn_relu3d_m = ConvBnRelu(512,512, 3, 1,3 // 2)
        self.conv_bn_relu3d_2 = ConvBnRelu(512,256, 3, 1,3 // 2)

        self.conv_bn_relu2d_1 = ConvBnRelu(256+128,256, 3, 1,3 // 2)
        self.conv_bn_relu2d_m = ConvBnRelu(256,256, 3, 1,3 // 2)
        self.conv_bn_relu2d_2 = ConvBnRelu(256,128, 3, 1,3 // 2)

        self.conv_bn_relu1d_1 = ConvBnRelu(128+64,128, 3, 1,3 // 2)
        self.conv_bn_relu1d_m = ConvBnRelu(128,128, 3, 1,3 // 2)
        self.conv_bn_relu1d_2 = ConvBnRelu(128,64, 3, 1,3 // 2)

        ## -------------Bilinear Upsampling--------------
        # self.upscore6 = nn.Upsample(scale_factor=8,mode='bilinear')###
        # self.upscore5 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        ## -------------Side Output--------------
        # self.outconvb = nn.Conv2d(512,1,3,padding=1)
        # self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        # self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(512,1,3,padding=1)
        self.outconv3 = nn.Conv2d(256,1,3,padding=1)
        self.outconv2 = nn.Conv2d(128,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)
    
    def forward(self,x):
        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)#1
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx) # torch.Size([1, 64, 224, 224])#2-7
        h2 = self.encoder2(h1) # torch.Size([1, 128, 112, 112])  #8-16  
        h3 = self.encoder3(h2) # torch.Size([1, 256, 56, 56])#17-29
        h4 = self.encoder4(h3) # torch.Size([1, 512, 28, 28])#30-36

        hbg = self.aspp(h4)       #37-42 

        # hx = hd5
        hx = hbg

        hx = self.conv_bn_relu4d_1(torch.cat((hx,h4),1))#43
        hx = self.conv_bn_relu4d_m(hx)#44
        hd4 = self.conv_bn_relu4d_2(hx)#45

        hx = self.upscore2(hd4) # 32 -> 64

        hx = self.conv_bn_relu3d_1(torch.cat((hx,h3),1))#46
        hx = self.conv_bn_relu3d_m(hx)#47
        hd3 = self.conv_bn_relu3d_2(hx)#48

        hx = self.upscore2(hd3) # 64 -> 128

        hx = self.conv_bn_relu2d_1(torch.cat((hx,h2),1))#49
        hx = self.conv_bn_relu2d_m(hx)#50
        hd2 = self.conv_bn_relu2d_2(hx)#51

        hx = self.upscore2(hd2) # 128 -> 256

        hx = self.conv_bn_relu1d_1(torch.cat((hx,h1),1))#52
        hx = self.conv_bn_relu1d_m(hx)#53
        hd1 = self.conv_bn_relu1d_2(hx)#54

        #  ------------- edge side -------------
        hx = self.conv1_1_down(h1)#55
        hx = self.conv1_2_down(hx)#56-57
        h1e = self.conv1_3_down(hx)#58


        hx = self.conv2_1_down(h2)#59
        hx = self.conv2_2_down(hx)#60-61
        h2e = self.conv2_3_down(hx)#62
        h2e = self.upscore2e(h2e)

        hx = self.conv3_1_down(h3)#63
        hx = self.conv3_2_down(hx)#64-65
        h3e = self.conv3_3_down(hx)#66
        h3e = self.upscore3e(h3e)

        hx = self.conv4_1_down(h4)#67
        hx = self.conv4_2_down(hx)#68-69
        h4e = self.conv4_3_down(hx)#70
        h4e = self.upscore4e(h4e)

        hx = torch.cat((h1e,h2e),1)
        hx = torch.cat((hx,h3e),1)
        hx = torch.cat((hx,h4e),1)

        hx = self.edgev(hx)#71
        fedge = self.edgef(hx)#72

        ## -------------Side Output-------------

        # dscnn = self.outconv6(s_2)
        # dscnn = self.upscore5(dscnn)

        # d5 = self.outconv5(hd5)
        # d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)#73
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)#74
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2) #75
        d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256    #76

        return  d1, d2, d3,d4, fedge