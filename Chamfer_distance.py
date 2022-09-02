# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import torch
import torch.nn as nn

########################################
##   Functions
########################################

def batch_pairwise_distance_matrix(X, Y, use_squared_distance=True):
    ''' [Bx]NxD, [Bx]MxD -> [Bx]NxM '''
    b1, n, d1 = X.size()
    b2, m, d2 = Y.size()
    assert(b1==b2)
    assert(d1==d2)
    X2= (X**2).sum(-1).unsqueeze(-1) #BxNx1
    Y2= (Y**2).sum(-1).unsqueeze(-2) #Bx1xM
    C = torch.bmm(X, Y.transpose(-1,-2)) #BxNxM
    D = X2 + Y2 - 2.0*C

    if not use_squared_distance:
        D = torch.sqrt(D+1e-8)

    return D


def chamfer_distance(D):
    ''' BxNxM -> B '''

    Dr = D.min(dim=-1)[0].mean(dim=-1, keepdim=True) #mean(min distance per row), Bx1
    Dc = D.min(dim=-2)[0].mean(dim=-1, keepdim=True) #mean(min distance per column), Bx1

    return torch.cat((Dr,Dc), dim=-1).sum(dim=-1) #1


## Date: 2/21/19 ###################################
def chamfer_distance_1_direction(D):
    ''' BxNxM -> B '''

    Dr = D.min(dim=-1)[0].mean(dim=-1, keepdim=True) #mean(min distance per row), Bx1
    # Dc = D.min(dim=-2)[0].mean(dim=-1, keepdim=True) #mean(min distance per column), Bx1

    return Dr

def chamfer_distance_mean(D):
    ''' BxNxM -> B '''

    Dr = D.min(dim=-1)[0].mean(dim=-1, keepdim=True) #mean(min distance per row), Bx1
    Dc = D.min(dim=-2)[0].mean(dim=-1, keepdim=True) #mean(min distance per column), Bx1

    return torch.cat((Dr,Dc), dim=-1).mean(dim=-1) #1

def chamfer_distance_max(D):
    ''' BxNxM -> B '''

    Dr = D.min(dim=-1)[0].mean(dim=-1, keepdim=True) #mean(min distance per row), Bx1
    Dc = D.min(dim=-2)[0].mean(dim=-1, keepdim=True) #mean(min distance per column), Bx1

    return torch.cat((Dr,Dc), dim=-1).sum(dim=-1)[0] #1


def hausdorff_distance(D):
    ''' BxNxM -> B '''
    Drc = D.min(dim=-1)[0].max(dim=-1,keepdim=True)[0] #Bx1, max(min distance per row)
    Dcr = D.min(dim=-2)[0].max(dim=-1,keepdim=True)[0] #Bx1, max(min distance per column)
    return torch.cat((Drc,Dcr), dim=-1).max(dim=-1)[0] #B

def chamfer_distance_sumknn(Dcd, X, Y, kn, Dy):

    from torch.nn.functional import softmin

    #Changed the mean of the chamfer distance to sum (original definition)
    Dr, indr = Dcd.min(dim=-1)
    Dr = Dr.mean(dim=-1, keepdim=True) #sum(min distance per row), Bx1
    Dc, indc = Dcd.min(dim=-2)
    Dc = Dc.mean(dim=-1, keepdim=True)  #sum(min distance per column), Bx1
    # smin = softmin(Dcd/100, -2) #temperature = 100

    smin = torch.zeros(Dcd.shape)
    for idx in range(smin.shape[2]):
        smin[0, indc[0, idx], idx] = 1.0

    smin = smin.cuda()

    Dknn = knn_distance(X, Y, smin, kn, Dy).sum(-1) #Try mean

    return torch.cat((Dr,Dc), dim=-1).sum(dim=-1), Dknn#1

def knn_distance(X, Y, smin, kn, Dy):

    #X - Estimated PC
    #Y - Original PC
    k = 4
    XX  = torch.bmm(smin.transpose(1,2), X)
    Dx = batch_pairwise_distance_matrix(XX, XX)
    Dy = batch_pairwise_distance_matrix(Y, Y)
    kn = Dy.topk(k=k, dim=1, largest=False)[1]

    dists_x = torch.gather(Dx, 1, kn)
    dists_y = torch.gather(Dy, 1, kn)

    D = dists_x - dists_y
    D = (D**2)[:,1:].sum(1)

    # print(D.cpu())

    return D


########################################
##   Classes
########################################
class BatchPairwiseDistanceMatrix(nn.Module):
    ''' BxNxD, BxMxD -> BxNxM '''

    def __init__(self, use_squared_distance=True):
        super(BatchPairwiseDistanceMatrix, self).__init__()
        self.use_squared_distance = use_squared_distance

    def forward(self, X, Y):
        return batch_pairwise_distance_matrix(
            X, Y,
            use_squared_distance=self.use_squared_distance)


class ChamferDistance(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, *args, **kwargs):
        super(ChamferDistance, self).__init__(*args, **kwargs)

    def forward(self, X, Y):
        return chamfer_distance(
            super(ChamferDistance, self).forward(X, Y) #Bxnxm
        )


class ChamferDistance_1_direction(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, *args, **kwargs):
        super(ChamferDistance_1_direction, self).__init__(*args, **kwargs)

    def forward(self, X, Y):
        return chamfer_distance_1_direction(
            super(ChamferDistance_1_direction, self).forward(X, Y) #Bxnxm
        )

class ChamferDistance_mean(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, *args, **kwargs):
        super(ChamferDistance_mean, self).__init__(*args, **kwargs)

    def forward(self, X, Y):
        return chamfer_distance_mean(
            super(ChamferDistance_mean, self).forward(X, Y) #Bxnxm
        )

class ChamferDistance_max(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, *args, **kwargs):
        super(ChamferDistance_max, self).__init__(*args, **kwargs)

    def forward(self, X, Y):
        return chamfer_distance_max(
            super(ChamferDistance_max, self).forward(X, Y) #Bxnxm
        )

class HausdorffDistance(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, *args, **kwargs):
        super(HausdorffDistance, self).__init__(*args, **kwargs)

    def forward(self, X, Y):
        return hausdorff_distance(
            super(HausdorffDistance, self).forward(X, Y) #Bxnxm
        )


class HybridChamferHausdorffDistance(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, chamfer_weight=0.5, **kwargs):
        super(HybridChamferHausdorffDistance, self).__init__(**kwargs)
        assert(0.<chamfer_weight<1)
        self.chamfer_weight = chamfer_weight

    def forward(self, X, Y):
        D = super(HybridChamferHausdorffDistance, self).forward(X, Y) #Bxnxm
        return (1.-self.chamfer_weight) * hausdorff_distance(D)\
               + self.chamfer_weight * chamfer_distance(D)

class ChamferDistance_sumknn(BatchPairwiseDistanceMatrix):
    ''' BxNxD, BxMxD -> B '''

    def __init__(self, *args, **kwargs):
        super(ChamferDistance_sumknn, self).__init__(*args, **kwargs)

    def forward(self, X, Y, kn, Dy):
        return chamfer_distance_sumknn(
            super(ChamferDistance_sumknn, self).forward(X, Y), X, Y, kn, Dy #Bxnxm
        )
