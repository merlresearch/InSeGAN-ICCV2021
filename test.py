#!/usr/bin/env python3
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import args
import torch
from compute_iou import compute_iou
from PIL import ImageFilter
from PIL import Image

def normalize_instance_segments(single_insts):
    # xx is assumed [num_inst] bs x 1 x 64 x 64
    for t in range(len(single_insts)):
        xx = single_insts[t]
        sz = xx.shape[-1]
        xx = filter_instance(xx)
        single_insts[t] = xx.reshape(xx.shape[0], 1, sz, sz)
    return single_insts

def filter_instance(xx):
    xx = xx.repeat(1,3,1,1).permute(0,2,3,1).data.cpu().numpy()
    for t in range(len(xx)):
        yy = xx[t]
        yy[yy<-args.depth_threshold]=0
        yy = Image.fromarray((yy*255/max(1,yy.max())).astype('uint8'))
        yy = yy.filter(ImageFilter.MedianFilter(3))
        yy = yy.filter(ImageFilter.GaussianBlur(2))
        xx[t] = yy

    xx = torch.Tensor(xx).permute(0,3,1,2).mean(1).unsqueeze(1)
    return xx


def find_instance_segments(single_insts, depth_theshold_for_inference=0.9):
    # generate the instance segmentations.
    single_insts = normalize_instance_segments(single_insts)
    depth_thresh = -1.*depth_theshold_for_inference
    combined_insts=torch.cat([ss.unsqueeze(4) for ss in single_insts], dim=4)
    valid_idx=combined_insts.argmax(4)
    ss = single_insts[0]<= depth_thresh #single_insts[0].mean()
    for kk in range(args.num_inst): ss = ss.mul(single_insts[kk] <= args.depth_threshold) # its <=-0.9 default.
    valid_idx[ss]=-1

    inst_seg = torch.zeros(3, *single_insts[0].shape); #insta_seg = inst_seg.unsqueeze(4).repeat(1,1,1,1,3)
    for kk in range(args.num_inst):
        for ii in range(3):
            inst_seg[ii, valid_idx == kk] = args.colors[kk,ii].float()

    inst_seg=inst_seg.squeeze(2).transpose(1,0)
    return inst_seg

def validation_error(test_loader, netG, netE):
    L1criterion = torch.nn.L1Loss()
    test_error = 0.
    iou = 0.
    with torch.no_grad():
        for i, (inputs, gt) in enumerate(test_loader):
            inputs = inputs.cuda()
            z, _ = netE(inputs)
            z = z.transpose(2,1)
            decoded_fake = [None] * args.num_inst
            if True: #num_inst > 1:
                fake = 0
                for t in range(args.num_inst):
                    fake_inst = netG.dec1(z[:,:,t].unsqueeze(2).unsqueeze(2))
                    fake = fake + fake_inst
                    decoded_fake[t] = netG.dec2(fake_inst)

                inst_seg = find_instance_segments(decoded_fake, args.depth_threshold)
                outputs = netG.dec2(fake/float(args.num_inst))
            elif args.num_inst == 1:
                z = z.unsqueeze(3)
                fake = netG.dec1(z)
                outputs = netG.dec2(fake)
            test_error = test_error + L1criterion(outputs, inputs)
            iou += compute_iou(gt, inst_seg)
        test_error = test_error/float(len(test_loader))
        miou = iou/float(len(test_loader))
    return test_error, miou

def encoder_decoder(inputs, netG, netE):
    decoded_fake = [None] * args.num_inst
    with torch.no_grad():
        z, _ = netE(inputs)
        z = z.transpose(2,1)
        if args.num_inst > 1:
            fake = 0
            for t in range(args.num_inst):
                fake_inst = netG.dec1(z[:,:,t].unsqueeze(2).unsqueeze(2))
                fake = fake + fake_inst
                decoded_fake[t] = netG.dec2(fake_inst)
            outputs = netG.dec2(fake/float(args.num_inst))
        elif args.num_inst == 1:
            z = z.unsqueeze(3)
            fake = netG.dec1(z)
            outputs = netG.dec2(fake)
            decoded_fake[0] = outputs

    return inputs, outputs, decoded_fake
