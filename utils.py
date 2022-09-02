#!/usr/bin/env python3
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import args
import torch
import numpy as np
from itertools import permutations
from Chamfer_distance import batch_pairwise_distance_matrix

if args.alignment == 'OT':
    from joblib import Parallel, delayed
    from ipot import sinkhorn_stabilized

colors = torch.randint(0,255, (args.num_inst*3,)).reshape(args.num_inst,3)

perms = torch.tensor(list(permutations(range(args.num_inst))))
print('---------- num instances = %d -----------'%(args.num_inst))

get_alignment = lambda X,Y: batch_pairwise_distance_matrix(X, Y).min(dim=2)[1]

batched_index_select = lambda X, idx: torch.cat([X[t,idx[t],:].unsqueeze(0).unsqueeze(0) for t in range(len(X))], dim=0)

get_matching_permutation = lambda D: perms[torch.tensor([torch.tensor([D[ii, perms[t][ii]]
                    for ii in range(args.num_inst)]).sum() for t in range(len(perms))]).argmin()]


def get_rotation_matrices(num_angles=12):
    theta = np.zeros((num_angles*2*3,3));
    theta[:num_angles*2,0] = np.arange(-np.pi, np.pi,np.pi/num_angles)
    theta[num_angles*2:num_angles*4,1] = np.arange(-np.pi, np.pi,np.pi/num_angles)
    theta[num_angles*4:,2] = np.arange(-np.pi, np.pi,np.pi/num_angles)
    return theta

def compute_alignment(z_true, z_pred, alignment='OT'):
    """
    computes optimal transport between z_true features and z_pred features
    expects z to be B x N x d where B is batch size, N is the number of features, and d is the feature dim.

    We also have alignments using either a greedy approach (alignment='greedy') or
    using an enumeration of all permutations (in case that is not many), i.e., alignment='enumerate'
    """

    batch_size, N, _ = z_true.shape
    marginal_Zt = np.ones(N)/N
    marginal_Zp = np.ones(N)/N
    Pi = torch.zeros((batch_size, N), dtype=int)
    D = batch_pairwise_distance_matrix(z_true, z_pred)
    D = [D[t]/(D[t].max()+1e-10) for t in range(len(D))]

    def greedy(dd):
        ii = torch.zeros((dd.shape[0],),dtype=int)
        for t in range(dd.shape[0]):
            ii[t] = dd[t].argmin()
            dd[:,ii[t]] = float('inf')
        return ii

    def sinkhorn(dd):
        # this uses iPOT solver.
        return sinkhorn_stabilized(marginal_Zp, marginal_Zt, dd, 0.0001, numItermax=1000, tau=1e3, stopThr=1e-3, print_period=1)

    if alignment == 'greedy':
        for t in range(batch_size):
            Pi[t]=greedy(D[t])

    if alignment == 'enumerate':
        for t in range(batch_size):
            Pi[t] = get_matching_permutation(D[t])

    if alignment == 'OT':
        if D[0].shape[1] == 2:
            Pi = torch.zeros(batch_size, 2)
            for t in range(batch_size):
                if D[t][0,0] + D[t][1,1] < D[t][1,0] + D[t][0,1]:
                    Pi[t,:] = torch.tensor([0, 1])
                else:
                    Pi[t,:] = torch.tensor([1, 0])
            Pi = Pi.long()
        else:
            if args.parallel is None:
                args.parallel = Parallel(n_jobs=1) #, prefer='threads')

            softPi = args.parallel(delayed(sinkhorn)(D[t]) for t in range(batch_size))
            Pi = torch.tensor(softPi).argmax(2)

    return Pi
