#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 2019/5/16 下午5:30 
  description: pose estimation 6D vector [tx,ty,tz,rx,ry,rz]
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_


# The down sample block
def conv(in_ch,out_ch,kernel_size=3):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size,stride=2,padding=(kernel_size-1)//2),
                         nn.ReLU(inplace=True))
# The up sample block
def up_conv(in_ch, out_ch):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                         nn.ReLU(inplace=True))  # out_padding ensure taht up sample >= down sample (in case odd layer)

class PoseNet(nn.Module):
    def __init__(self,nb_ref_imgs=2,output_exp=False):
        super(PoseNet,self).__init__()
        self.nb_ref_imgs=nb_ref_imgs
        self.output_exp=output_exp
        out_chs = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = conv(3*(1+nb_ref_imgs),out_chs[0],kernel_size=7)  # 32*208*64
        self.conv2 = conv(out_chs[0], out_chs[1], kernel_size=5) # 64*104*32
        self.conv3 = conv(out_chs[1], out_chs[2])  # 128*52*16
        self.conv4 = conv(out_chs[2], out_chs[3])  # 256*26*8
        self.conv5 = conv(out_chs[3], out_chs[4])  # 512*13*4
        self.conv6 = conv(out_chs[4], out_chs[5])  # 512*6*2
        self.conv7 = conv(out_chs[5], out_chs[6])  # 512*3*1

        self.pose_pred=nn.Conv2d(out_chs[6], 6*self.nb_ref_imgs,kernel_size=1)  #12*3*1

        up_chs=[256,128,64,32,16]
        self.up_conv5=up_conv(out_chs[4],up_chs[0])
        self.up_conv4 = up_conv(up_chs[0], up_chs[1])
        self.up_conv3 = up_conv(up_chs[1], up_chs[2])
        self.up_conv2 = up_conv(up_chs[2], up_chs[3])
        self.up_conv1 = up_conv(up_chs[3], up_chs[4])

        self.predict_mask4 = nn.Conv2d(up_chs[1], self.nb_ref_imgs, kernel_size=3, padding=1)
        self.predict_mask3 = nn.Conv2d(up_chs[2], self.nb_ref_imgs, kernel_size=3, padding=1)
        self.predict_mask2 = nn.Conv2d(up_chs[3], self.nb_ref_imgs, kernel_size=3, padding=1)
        self.predict_mask1 = nn.Conv2d(up_chs[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, tgt_img, rel_imgs):
        assert (len(rel_imgs) == self.nb_ref_imgs)
        x = [tgt_img]
        x.extend(rel_imgs)
        x = torch.cat(x,1)
        out_conv1=self.conv1(x)
        out_conv2=self.conv2(out_conv1)
        out_conv3=self.conv3(out_conv2)
        out_conv4=self.conv4(out_conv3)
        out_conv5=self.conv5(out_conv4)
        out_conv6=self.conv6(out_conv5)
        out_conv7=self.conv7(out_conv6)
        pose=self.pose_pred(out_conv7)
        pose=pose.mean(3).mean(2)  # 12 dim=3 calculate mean; dim=2 calculate mean
        pose=0.01*pose.view(pose.size[0],self.nb_ref_imgs,6)  #(nb_ref_imgs,6)
        if self.output_exp:
            out_up_conv5 = self.up_conv5(out_conv5)
            out_up_conv4 = self.up_conv4(out_up_conv5)
            out_up_conv3 = self.up_conv3(out_up_conv4)
            out_up_conv2 = self.up_conv2(out_up_conv3)
            out_up_conv1 = self.up_conv1(out_up_conv2)

            exp_mask4 = torch.sigmoid(self.predict_mask4(out_up_conv4))
            exp_mask3 = torch.sigmoid(self.predict_mask3(out_up_conv3))
            exp_mask2 = torch.sigmoid(self.predict_mask2(out_up_conv2))
            exp_mask1 = torch.sigmoid(self.predict_mask1(out_up_conv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None
        if self.training:
            return [exp_mask1,exp_mask2,exp_mask3,exp_mask4], pose
        else:
            return exp_mask1,pose

