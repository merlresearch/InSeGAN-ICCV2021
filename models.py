#!/usr/bin/env python3
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from torch import nn
import args
from Hologan import HoloNet as hologan_implicit

## Pose Encoder ####
class Encoder(nn.Module):
    def __init__(self, ngpu, num_inst):
        super(Encoder, self).__init__()
        self.ngpu = args.ngpu
        self.z_dim = args.nz
        self.ngf = args.ngf
        self.ndf = args.ndf
        self.nc = args.nc
        self.num_inst = args.num_inst

        self.enc_main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 2),
            nn.InstanceNorm2d(self.ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
        )

        self.enc_sub = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 4),
            nn.InstanceNorm2d(self.ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(self.ndf * 8, self.z_dim*num_inst, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        encoded_input = self.enc_main(input)
        z = self.enc_sub(encoded_input).reshape(input.shape[0],self.z_dim, args.num_inst).transpose(2,1)
        return z, encoded_input

## Depth Image Generator ####
class Generator(nn.Module):
    # generator from multiple z to multiple instances.
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = args.ngpu
        self.z_dim = 6 # this is the axis angle vector dimension (3 for rot and 3 for trans.)
        self.ngf = args.ngf
        self.ndf = args.ndf
        self.nc = args.nc
        self.nz = args.nz
        self.num_arch = args.num_archs

        self.dec1 = hologan_implicit(self.ngf, out_ch=1, z_dim=self.nz) # ngf*2 x 16 x 16

        self.dec2 = nn.Sequential(
          # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.InstanceNorm2d(self.ngf, affine=True),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
            )

    def forward(self, z):
        inputs = 0
        for t in range(len(z)):
            inputs += self.dec1(z[t])
        inputs = inputs/float(len(z))
        return self.dec2(inputs)

## Discriminator ####

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu
        self.ndf = args.ndf
        self.nc = args.nc
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 2),
            nn.InstanceNorm2d(self.ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 4),
            nn.InstanceNorm2d(self.ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 8),
            nn.InstanceNorm2d(self.ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
