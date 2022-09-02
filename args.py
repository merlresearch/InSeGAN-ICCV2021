#!/usr/bin/env python3
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import argparse
import random
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='/homes/cherian/train_data/instance_seg/Anoops-Data', help='where is the data?')
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--inst', type=int, default=5, help='number of instances in ground truth (default=5)')
    parser.add_argument('--resume', action='store_true', help='restart from a checkpoint?')
    parser.add_argument('--trail', type=int, default=0)
    parser.add_argument('--obj', type=str, required=True, help='name of the object instance')
    parser.add_argument('--nz', type=int, default=128, help='latent space size of pose')
    parser.add_argument('--align', type=str, default='enumerate', help='type of pose alignment to use?')
    parser.add_argument('--dthresh', type=float, default=-0.2, help='threshold to use at inference in [-1,0], (default=-0.2)')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default 0.0002)')
    parser.add_argument('--nf', type=int, default=64, help='number of filters in the CNNs (default=64)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size to use (default 128)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam (default 0.5)')
    parser.add_argument('--source_seed', type=int, default=-1, help='if you restart training from another seed, what is that?')
    parser.add_argument('--no_EL3', action='store_true', help='do not use intermim feature alignment')
    parser.add_argument('--no_EL2', action='store_true', help='do not pose alignment')
    parser.add_argument('--test', action='store_true', help='do test evaluation?')
    parser.add_argument('--data_inst', type=int, default=5, help='number of instances in training data to assume')
    parser.add_argument('--num_workers', type=int, default=4, help='number of worker processes for training data (default 4)')

    args = parser.parse_args()
    return args

args = parse_args()
print(args)

seed = args.seed
if seed == 0:
    seed = random.randint(1, 10000)
seed = str(seed)

if args.source_seed == -1:
    args.source_seed = args.seed

args.ngf = args.nf
args.ndf = args.nf

obj_name = args.obj

experiment = './results/instagan/' + obj_name + '/instagan_' + seed
print(experiment)
if not os.path.exists(experiment):
    os.system('mkdir -p %s'%(experiment))
    os.system('mkdir -p %s'%(os.path.join(experiment, 'all/inputs_and_generated_samples/')))
    os.system('mkdir -p %s'%(os.path.join(experiment, 'all/single_generated_samples/')))
    os.system('mkdir -p %s'%(os.path.join(experiment, 'all/inputs_and_decoded_inputs/')))
    os.system('mkdir -p %s'%(os.path.join(experiment, 'all/holonet_inputs_and_decoded_inputs/')))

print('working on %s with %d instances: seed=%s'%(args.obj, args.inst, args.seed))

# number of instances to use
num_inst = args.inst;
pix_num=1024*num_inst

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are using GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True

# num_archs
num_archs = args.nz #int(sys.argv[1])

# Number of workers for dataloader
num_workers = args.num_workers

# Batch size during training
batch_size = args.batch_size

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = num_archs

# Size of feature maps in generator
ngf = args.nf

# Size of feature maps in discriminator
ndf = args.nf

update_freq = 10

# Number of training epochs
num_epochs = 3000

# Learning rate for optimizers
lr = args.lr

# Beta1 hyperparam for Adam optimizers
beta1 = args.beta1

# the code runs only on a single GPU. No CPU option is currently supported,
# but you may remove the .cuda() calls to make it run on CPU
ngpu = 1

trail_id = args.trail

alignment = args.align # type of alignment to use OT/enumerate/greedy.

depth_threshold = args.dthresh

parallel = None # If needed, we may use parallel threads for optimal transport (see utils.py)

resume = args.resume

test = args.test

no_EL3 = args.no_EL3

no_EL2 = args.no_EL2

print_freq = 1

source_seed = args.source_seed

colors = torch.randint(0,255, (num_inst*3,)).reshape(num_inst,3)

#root_dir='/homes/cherian/train_data/instance_seg/Goncalo_data/%s/'%(args.obj)
root_dir = os.path.join(args.root_dir, '%s'%(args.obj))
root_path = os.path.join(root_dir, 'depth')
val_path = os.path.join(root_dir, 'val/depth/')
gt_val_path = os.path.join(root_dir, 'val/rgb/')
test_path = os.path.join(root_dir, 'test/depth/')
gt_test_path = os.path.join(root_dir, 'test/rgb/')
print(root_dir)
