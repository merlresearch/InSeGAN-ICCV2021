#!/usr/bin/env python3

# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

def compute_iou(X, Y):
    """
    computes the best segmentation alignment between the predictions and the GT,
    and uses the best match to compute the IOU.
    """
    def compute_inst_tensor(insts):
        uq = torch.unique(insts)
        inst_map = []

        for t in range(len(uq)):
            if uq[t] == 0:
                continue
            inst_map.append((insts == uq[t]).unsqueeze(0))
        inst_map = torch.cat(inst_map, dim=0)

        return inst_map

    miou = 0.
    num_test = len(X)
    for t in range(num_test):
        im1 = X[t][0]
        im2 = Y[t][0]
        im2[im1 == 0] = 0 # remove points that are not in the ground truth, as depth will be -1 for those.

        try:
            gt = compute_inst_tensor(im1)
            pred = compute_inst_tensor(im2)
        except:
            print('error in compute_iou! ')
            continue

        # compute a matrix of best IoUs for every predicted instance against every gt instance
        # and use the best match, and average the IoU scores on all instances and all images.
        iou = torch.zeros(len(gt), len(pred))
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou[i,j] = ((gt[i]*pred[j])>0).sum().float()/(((gt[i]+pred[j])>0).sum().float() + 1e-5)
        miou += iou.max(dim=1)[0].mean()
    miou /= num_test
    return miou
