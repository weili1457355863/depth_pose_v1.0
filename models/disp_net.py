#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 2019/5/16 上午11:50 
  description: DispNet architecture that is mainly based on encoder-decoder design with skip connection and multi-scale
  side predictions(Input image 416*128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# The down sample block
def down_sample_conv(in_ch,out_ch,kernel_size=3):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size,stride=2,padding=(kernel_size-1)//2),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_ch,out_ch,kernel_size,stride=1,padding=(kernel_size-1)//2),
                         nn.ReLU(inplace=True))

# The down sample block
def up_conv(in_ch, out_ch):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                         nn.ReLU(inplace=True))  # out_padding ensure taht up sample >= down sample (in case odd layer)

# ensure the same size
def crop_like(input,ref):
    assert(input.size[2] >= ref.size[2] and input.size[3] >= ref.size[3])
    return input[:, :, :ref.size[2], :ref.size[3]]

# The down sample block
def conv(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1),
                         nn.ReLU(inplace=True))

# predict disparity
def predict_disparity(in_ch):
    return nn.Sequential(nn.Conv2d(in_ch,1,kernel_size=3,padding=1),
                         nn.Sigmoid())

class DispNet(nn.Module):
    def __init__(self, alpha=10, beta=0.01):
        super(DispNet, self).__init__()
        self.alpha=alpha
        self.beta=beta
        # encoder
        down_chs=[32, 64, 128, 256, 512, 512, 512]
        self.down_conv1 = down_sample_conv(3,down_chs[0],kernel_size=7)  # 3*416*128->32*208*64
        self.down_conv2 = down_sample_conv(down_chs[0], down_chs[1], kernel_size=5)  # 64*104*32
        self.down_conv3 = down_sample_conv(down_chs[1], down_chs[2], kernel_size=5)  # 128*52*16
        self.down_conv4 = down_sample_conv(down_chs[2], down_chs[3], kernel_size=5)  # 256*26*8
        self.down_conv5 = down_sample_conv(down_chs[3], down_chs[4], kernel_size=5)  # 512*13*4
        self.down_conv6 = down_sample_conv(down_chs[4], down_chs[5], kernel_size=5)  # 512*6*2
        self.down_conv7 = down_sample_conv(down_chs[5], down_chs[6], kernel_size=5)  # 512*3*1
        # decoder
        up_chs=[512, 512, 256, 128, 64, 32, 16]
        self.up_conv1 = up_conv(down_chs[6], up_chs[0])
        self.up_conv2 = up_conv(up_chs[0], up_chs[1])
        self.up_conv3 = up_conv(up_chs[1], up_chs[2])
        self.up_conv4 = up_conv(up_chs[2], up_chs[3])
        self.up_conv5 = up_conv(up_chs[3], up_chs[4])
        self.up_conv6 = up_conv(up_chs[4], up_chs[5])
        self.up_conv7 = up_conv(up_chs[5], up_chs[6])

        self.iconv1 = conv(up_chs[0] + down_chs[5],up_chs[0])
        self.iconv2 = conv(up_chs[1] + down_chs[4], up_chs[1])
        self.iconv3 = conv(up_chs[2] + down_chs[3], up_chs[2])
        self.iconv4 = conv(up_chs[3] + down_chs[2] + 1, up_chs[3])
        self.iconv5 = conv(up_chs[4] + down_chs[1] + 1, up_chs[4])
        self.iconv6 = conv(up_chs[5] + down_chs[0] + 1, up_chs[5])
        self.iconv7 = conv(up_chs[6] + 1, up_chs[6])

        self.predict_dispiraty1 = predict_disparity(up_chs[3])
        self.predict_dispiraty2 = predict_disparity(up_chs[4])
        self.predict_dispiraty3 = predict_disparity(up_chs[5])
        self.predict_dispiraty4 = predict_disparity(up_chs[6])
    def forward(self, x):
        out_conv1 = self.down_conv1(x)
        out_conv2 = self.down_conv2(out_conv1)
        out_conv3 = self.down_conv3(out_conv2)
        out_conv4 = self.down_conv4(out_conv3)
        out_conv5 = self.down_conv5(out_conv4)
        out_conv6 = self.down_conv6(out_conv5)
        out_conv7 = self.down_conv7(out_conv6)

        out_upconv7=crop_like(self.up_conv1(out_conv7),out_conv6)
        concate7=torch.cat((out_upconv7,out_conv6),1)  # skip_connection dim=1 C
        out_iconv7=self.iconv1(concate7)  # yield smoother disparity

        out_upconv6=crop_like(self.up_conv2(out_iconv7),out_conv5)
        concate6=torch.cat((out_upconv6,out_conv5),1)
        out_iconv6=self.iconv1(concate6)

        out_upconv5=crop_like(self.up_conv3(out_iconv6),out_conv4)
        concate5=torch.cat((out_upconv5,out_conv4),1)
        out_iconv5=self.iconv1(concate5)

        out_upconv4=crop_like(self.up_conv4(out_iconv5),out_conv3)
        concate4=torch.cat((out_upconv4,out_conv3),1)
        out_iconv4=self.iconv1(concate4)
        disp4= self.alpha*self.predict_disparity(out_iconv4)+self.beta

        out_upconv3=crop_like(self.up_conv5(out_iconv4),out_conv2)
        up_disp4=crop_like(F.interpolate(disp4,scale_factor=2,mode='bilinear',align_corners=False),out_conv2)
        concate3=torch.cat((out_upconv3,out_conv2,up_disp4),1)
        out_iconv3=self.iconv1(concate3)
        disp3= self.alpha*self.predict_disparity(out_iconv3)+self.beta

        out_upconv2=crop_like(self.up_conv5(out_iconv3),out_conv1)
        up_disp3=crop_like(F.interpolate(disp3,scale_factor=2,mode='bilinear',align_corners=False),out_conv1)
        concate2=torch.cat((out_upconv2,out_conv1,up_disp3),1)
        out_iconv2=self.iconv1(concate2)
        disp2= self.alpha*self.predict_disparity(out_iconv2)+self.beta

        out_upconv1=self.up_conv5(out_iconv2)
        up_disp2=F.interpolate(disp2,scale_factor=2,mode='bilinear',align_corners=False)
        concate1=torch.cat((out_upconv1,up_disp2),1)
        out_iconv1=self.iconv1(concate1)
        disp1= self.alpha*self.predict_disparity(out_iconv1)+self.beta
        if self.training:
            return disp1,disp2,disp3,disp4
        else:
            return disp1


