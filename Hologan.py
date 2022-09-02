#!/usr/bin/env python3
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019, Christopher Beckham
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torchgeometry as tgm

class HoloTrans():
    #def __init__(self, angles=[-90,90,-180,180,-90,90], *args, **kwargs):
    def __init__(self, angles=[-20,20,-180,180,-20,20], *args, **kwargs):
        super(HoloTrans, self).__init__(*args, **kwargs)
        self.angles = self._angles_to_dict(angles)
        self.rot2idx = {
            'x': 0,
            'y': 1,
            'z': 2
        }

    def _to_radians(self, deg):
        return deg * (np.pi / 180)

    def _angles_to_dict(self, angles):
        angles = {
            'min_angle_x': self._to_radians(angles[0]),
            'max_angle_x': self._to_radians(angles[1]),
            'min_angle_y': self._to_radians(angles[2]),
            'max_angle_y': self._to_radians(angles[3]),
            'min_angle_z': self._to_radians(angles[4]),
            'max_angle_z': self._to_radians(angles[5])
        }
        return angles

    def rot_matrix_x(self, theta):
        """
        theta: measured in radians
        """
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = 1.
        mat[1, 1] = np.cos(theta)
        mat[1, 2] = -np.sin(theta)
        mat[2, 1] = np.sin(theta)
        mat[2, 2] = np.cos(theta)
        return mat

    def rot_matrix_y(self, theta):
        """
        theta: measured in radians
        """
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = np.cos(theta)
        mat[0, 2] = np.sin(theta)
        mat[1, 1] = 1.
        mat[2, 0] = -np.sin(theta)
        mat[2, 2] = np.cos(theta)
        return mat

    def rot_matrix_z(self, theta):
        """
        theta: measured in radians
        """
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = np.cos(theta)
        mat[0, 1] = -np.sin(theta)
        mat[1, 0] = np.sin(theta)
        mat[1, 1] = np.cos(theta)
        mat[2, 2] = 1.
        return mat

    def pad_rotmat(self, theta):
        """theta = (3x3) rotation matrix"""
        return np.hstack((theta, np.zeros((3,1))))

    def sample_angles(self,
                      bs,
                      min_angle_x,
                      max_angle_x,
                      min_angle_y,
                      max_angle_y,
                      min_angle_z,
                      max_angle_z):
        """Sample random yaw, pitch, and roll angles"""
        angles = []
        for i in range(bs):
            rnd_angles = [
                np.random.uniform(min_angle_x, max_angle_x),
                np.random.uniform(min_angle_y, max_angle_y),
                np.random.uniform(min_angle_z, max_angle_z),
            ]
            angles.append(rnd_angles)
        return np.asarray(angles)

    def get_theta(self, angles):
        '''Construct a rotation matrix from angles.
        This uses the Euler angle representation. But
        it should also work if you use an axis-angle
        representation.
        '''
        bs = len(angles)
        theta = np.zeros((bs, 3, 4))

        angles_x = angles[:, 0]
        angles_y = angles[:, 1]
        angles_z = angles[:, 2]
        for i in range(bs):
            theta[i] = self.pad_rotmat(
                np.dot(np.dot(self.rot_matrix_z(angles_z[i]), self.rot_matrix_y(angles_y[i])),
                       self.rot_matrix_x(angles_x[i]))
            )
            theta[i,:,3] = (np.random.rand(3,)-0.5)/2.

        return torch.from_numpy(theta).float()

    def sample(self, bs, seed=None):
        """Return a sample G(z)"""
        self._eval()
        with torch.no_grad():
            z_batch = self.sample_z(bs, seed=seed)
            angles = self.sample_angles(z_batch.size(0),
                                        **self.angles)
            thetas = self.get_theta(angles)
            if z_batch.is_cuda:
                thetas = thetas.cuda()
            gz = self.g(z_batch, thetas)
        return gz

    def _generate_rotations(self,
                            z_batch,
                            axes=['x', 'y', 'z'],
                            min_angle=None,
                            max_angle=None,
                            num=5):
        dd = dict()
        for rot_mode in axes:
            if min_angle is None:
                min_angle = self.angles['min_angle_%s' % rot_mode]
            if max_angle is None:
                max_angle = self.angles['max_angle_%s' % rot_mode]
            pbuf = []
            with torch.no_grad():
                for p in np.linspace(min_angle, max_angle, num=num):
                    #enc_rot = gan.rotate_random(enc, angle=p)
                    angles = np.zeros((z_batch.size(0), 3)).astype(np.float32)
                    angles[:, self.rot2idx[rot_mode]] += p
                    thetas = self.get_theta(angles)
                    if z_batch.is_cuda:
                        thetas = thetas.cuda()
                    x_fake = self.g(z_batch, thetas)
                    pbuf.append(x_fake*0.5 + 0.5)
            dd[rot_mode] = pbuf
        return dd

    def sample_transforms(self, bs):
        angles = self.sample_angles(bs, **self.angles)
        thetas = self.get_theta(angles)
        #angles_t = torch.from_numpy(angles).float().cuda()
        return thetas

class ResBlock2d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        self.bn = nn.InstanceNorm2d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_ch)

        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv2d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)

class ResBlock3d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock3d, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, padding=1)
        self.bn = nn.InstanceNorm3d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm3d(out_ch)

        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv3d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)

def _adain_module_3d(z_dim, out_ch):
    adain = nn.InstanceNorm3d(out_ch, affine=True)
    z_mlp = nn.Sequential(
        nn.Linear(z_dim, out_ch*2), # both var and mean
    )
    return adain, z_mlp

def _adain_module_2d(z_dim, out_ch):
    adain = nn.InstanceNorm2d(out_ch, affine=True)
    z_mlp = nn.Linear(z_dim, out_ch*2)
    return adain, z_mlp

class HoloNet(nn.Module):

    def __init__(self, nf, out_ch=3, z_dim=128):
        super(HoloNet, self).__init__()

        self.ups_3d = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups_2d = nn.Upsample(scale_factor=2, mode='nearest')

        self.angle_select = torch.tensor([0,1,2]).cuda()
        self.trans_select = torch.tensor([3,4,5]).cuda()

        self.z_dim = z_dim

        # pose encoder.
        self.pe = nn.Sequential(
                nn.Linear(z_dim, z_dim//2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(z_dim//2, z_dim//4, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(z_dim//4, 6, bias=False)
        )

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight.data, 1.)
        self.pe.apply(weights_init)

        xstart = ( torch.randn((1, nf, 4, 4, 4)) - 0.5 ) / 0.5
        nn.init.xavier_uniform_(xstart.data, 1.)
        self.xstart = nn.Parameter(xstart)
        self.xstart.requires_grad = True

        self.nf = nf

        self.rb1 = ResBlock3d(nf, nf // 2)
        self.adain_1, self.z_mlp1 = _adain_module_3d(z_dim, nf//2)

        self.rb2 = ResBlock3d(nf // 2, nf // 4)
        self.adain_2, self.z_mlp2 = _adain_module_3d(z_dim, nf//4)

        # The 3d transformation is done here.

        # Two convs (no adain) that bring it from nf//4 to nf//8
        self.postproc = nn.Sequential(
            nn.Conv3d(nf//4, nf//8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf//8, affine=True),
            nn.ReLU(),
            nn.Conv3d(nf//8, nf//8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf//8, affine=True),
            nn.ReLU()
        )

        # Then concatenation happens.

        pnf = (nf//8)*(4**2) # 512

        # TODO: should be 1x1
        self.proj = nn.Sequential(
            nn.Conv2d(pnf, pnf, kernel_size=3, padding=1),
            nn.InstanceNorm2d(pnf, affine=True),
            nn.ReLU()
        )

        self.tanh = nn.Tanh()
        self.im = 64

    def interpolate_trilinear(self, img, x, y, z):
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        x0 = torch.clamp(x0, min=0, max=img.shape[2] - 1)
        x1 = torch.clamp(x1, min=0, max=img.shape[2] - 1)
        y0 = torch.clamp(y0, min=0, max=img.shape[3] - 1)
        y1 = torch.clamp(y1, min=0, max=img.shape[3] - 1)
        z0 = torch.clamp(z0, min=0, max=img.shape[4] - 1)
        z1 = torch.clamp(z1, min=0, max=img.shape[4] - 1)


        x_ = x - x0.float()
        y_ = y - y0.float()
        z_ = z - z0.float()

        out = (img[:,:,x0,y0,z0]*(1-x_)*(1-y_)*(1-z_) +
                     img[:,:,x1,y0,z0]*x_*(1-y_)*(1-z_) +
                     img[:,:,x0,y1,z0]*(1-x_)*y_*(1-z_) +
                     img[:,:,x0,y0,z1]*(1-x_)*(1-y_)*z_ +
                     img[:,:,x1,y0,z1]*x_*(1-y_)*z_ +
                     img[:,:,x0,y1,z1]*(1-x_)*y_*z_ +
                     img[:,:,x1,y1,z0]*x_*y_*(1-z_) +
                     img[:,:,x1,y1,z1]*x_*y_*z_)
        return out

    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        out = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
        return out

    def _rshp2d(self, z):
        return z.view(-1, z.size(1), 1, 1)

    def _rshp3d(self, z):
        return z.view(-1, z.size(1), 1, 1, 1)

    def _split(self, z):
        len_ = z.size(1)
        mean = z[:, 0:(len_//2)]
        var = F.softplus(z[:, (len_//2):])
        return mean, var

    def compute_rot_matrix_by_qr_factorization(self, theta):
        theta = theta.reshape(-1,3,3)
        rot, triu = torch.qr(theta)
        return rot, triu

    def get_implicit_pose(self, z):

        pz = self.pe(z)
        theta = torch.index_select(pz, 1, self.angle_select)
        trans = torch.index_select(pz, 1, self.trans_select)
        rot = tgm.angle_axis_to_rotation_matrix(theta)
        rot = torch.index_select(torch.index_select(rot, 1, self.angle_select), 2, self.angle_select)
        return torch.cat([rot, trans.unsqueeze(2)], 2)

    def forward(self, z): #, thetas):#forward(self, z, thetas):
        if len(z.shape) == 4:
            z = z.squeeze(2).squeeze(2)
        thetas = self.get_implicit_pose(z)
        bs = len(thetas)#z.size(0)

        # (512, 4, 4, 4)
        xstart = self.xstart.repeat((bs, 1, 1, 1, 1))

        # (256, 8, 8, 8)
        h1 = self.adain_1(self.ups_3d(self.rb1(xstart)))

        # (128, 16, 16, 16)
        h2 = self.adain_2(self.ups_3d(self.rb2(h1)))

        # Perform rotation
        h2_rotated = self.stn(h2, thetas)

        # (64, 16, 16, 16)
        h4 = self.postproc(h2_rotated)

        # Projection unit. Concat depth and channels
        # (32*16, 16, 16) = (512, 16, 16)
        h4_proj = h4.view(-1, h4.size(1)*h4.size(2), h4.size(3), h4.size(4))

        # (256, 16, 16) (TODO: this should be a 1x1 conv)
        h4_proj = self.proj(h4_proj)
        return h4_proj
