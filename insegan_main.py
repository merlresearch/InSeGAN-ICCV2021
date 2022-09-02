#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import print_function

import args
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils
import numpy as np
import time

from dataset import ObjectDataset, ObjectDatasetTest, train_transforms, test_transforms, gt_transforms
from models import Encoder, Discriminator, Generator
from utils import *
import test

num_inst = args.num_inst
batch_size = args.batch_size
fake_gen_examples = None
######################################################################

def test_pose_encoder(test_loader, netE, netG, netH, netP):
    test_error = 0.
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            gen_inputs = netG(netE(inputs.cuda())[0][:,0,:].unsqueeze(0).unsqueeze(3).unsqueeze(3))
            outputs, _ = netH(None, netP(gen_inputs))
            test_error = test_error + L1criterion(outputs, gen_inputs)
    test_error = test_error/(float(len(test_loader))+1e-5)

    return test_error.data.item()

def save_model(net, acc, epoch, net_name, location, trail_id):
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
    if not os.path.isdir(location):
        os.mkdir(location)
    torch.save(state, os.path.join(location, 'ckpt_%s.pth'%(net_name)))


dataset = ObjectDataset(root=args.root_path, transforms_= train_transforms)
testset = ObjectDatasetTest(root=args.test_path, gtroot=args.gt_test_path, transforms_= test_transforms, gt_transforms_= gt_transforms)
valset = ObjectDatasetTest(root=args.val_path, gtroot=args.gt_val_path, transforms_= test_transforms, gt_transforms_= gt_transforms)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

def init_models(netG, netD, netE):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    assert((device.type == 'cuda') and (args.ngpu >= 1))

    netG.apply(weights_init)
    netE.apply(weights_init)
    netD.apply(weights_init)

    return netG, netD, netE

# Create the generator
netG = Generator(args.ngpu).to(device)
netD = Discriminator(args.ngpu).to(device)
netE = Encoder(args.ngpu, args.num_inst).to(device)
netG, netD, netE = init_models(netG, netD, netE)

# load pre-trained models
if args.resume or args.test:
    try:
        print('resuming from previous run')
        netG.load_state_dict(torch.load('./results/instagan/' + args.obj_name + '/instagan_' + str(args.source_seed) + '/ckpt_netG.pth')['net'])
        netD.load_state_dict(torch.load('./results/instagan/' + args.obj_name + '/instagan_' + str(args.source_seed) + '/ckpt_netD.pth')['net'])
        netE.load_state_dict(torch.load('./results/instagan/' + args.obj_name + '/instagan_' + str(args.source_seed) + '/ckpt_netE.pth')['net'])
    except:
        print('could not load all pre-trained models! training from scratch!')
######################################################################

# Initialize BCELoss function
criterion = nn.BCELoss()
L1criterion = nn.L1Loss()
MSEcriterion = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of training of the generator
fixed_input = torch.randn(num_inst, batch_size, args.nz,1,1).cuda()
input_samples = next(iter(testloader))[0].clone().cuda()
vutils.save_image(input_samples.detach().cpu().mul(0.5).add(0.5), '{0}/input_samples.png'.format(args.experiment))

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(list(netG.dec1.parameters())+list(netG.dec2.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=0.001, betas=(args.beta1, 0.999))

# do evaluation or test.
if args.test:
    val_error, miou = test.validation_error(testloader, netG, netE)
    print('obj=%s: miou = %f\n'%(args.obj_name, miou))
    inputs, outputs, single_insts = test.encoder_decoder(input_samples, netG, netE)

    input_outputs = torch.cat([inputs, outputs], dim=0)
    segments = test.find_instance_segments(single_insts, args.depth_threshold)
    ii = inputs.repeat(1,3,1,1)
    segments[ii == ii.min()] = 0
    input_outputs = torch.cat([input_outputs.repeat(1,3,1,1).detach().cpu().mul(0.5).add(0.5), segments/255], dim=0)
    input_outputs = torch.cat([input_outputs, torch.cat(single_insts, dim=0).repeat(1,3,1,1).detach().cpu().mul(0.5).add(0.5)], dim=0)
    nn = input_outputs.shape[0]//(num_inst+3)

    #np.save('{0}/best_inputs_and_decoded_inputs-{1}-{2}-qual-paper.npy'.format(args.experiment, args.seed, args.obj_name), input_outputs.numpy())
    test_result_file = '{0}/inputs_and_decoded_inputs-{1}-{2}-qual-paper.png'.format(args.experiment, args.seed, args.obj_name)
    vutils.save_image(input_outputs, test_result_file, padding=2,nrow=nn, pad_value=255)
    print('All test results saved to %s'%(test_result_file))

    quit()

# Training Loop
iters = 0
best_val_error = 10000.
best_pose_error = 10000.
best_miou = 0.

print("Starting Training Loop...")
for epoch in range(args.num_epochs):
    tt=time.time()
    D_x, D_G_z1, D_G_z2, tot_errD, tot_errG, tot_errE = 0., 0., 0., 0., 0., 0.
    na, nd, ng, nt, nal, ne = 0., 0., 0., 0., 0., 0.
    for i, data in enumerate(dataloader, 0):
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)

        # standard discriminator training.
        netD.zero_grad()

        output = netD(real_cpu).view(-1)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x += output.mean().item()

        ## Train with all-fake batch
        z = torch.randn(num_inst, b_size, args.nz, 1, 1).cuda()
        z_true = z

        # fake_interim is the intermediate fake image in latent space (single instance)
        fake_interim = netG.dec1(z[0].detach())
        for t in range(num_inst-1):
            fake_interim = fake_interim + netG.dec1(z[1+t].detach())
        fake = netG.dec2(fake_interim/float(num_inst)) # fake is the final generated multinstance fake image.

        fake_gen_examples = fake.detach()
        label = torch.full((fake.shape[0],), fake_label, dtype=torch.float, device=device)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 += output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        tot_errD += errD.item()
        # Update D
        optimizerD.step()
        nd += 1

        # update the generator.
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        tot_errG += errG.item()
        errG.backward()
        D_G_z2 += output.mean().item()
        optimizerG.step()
        ng += 1

        netE.zero_grad()

        z_true = z_true.squeeze(3).squeeze(3).transpose(1,0)
        z_pred, fake_pred_interim = netE(fake.detach())

        # intermediate reconstruction loss.
        errE_L1 = MSEcriterion(fake_pred_interim, fake_interim.detach())

        # this is the loss after pose decoding and generator re-encoding.
        if not args.no_EL3:
            zz = z_pred.transpose(1,0)
            fakefake_interim = netG.dec1(zz[0])
            for t in range(num_inst-1):
                fakefake_interim = fakefake_interim + netG.dec1(zz[1+t])
            fakefake = netG.dec2(fakefake_interim/float(num_inst))
            errE_L3 = MSEcriterion(fakefake, fake.detach())

        # pose alignment loss
        if not args.no_EL2:
            if num_inst == 1:
                errE_L2 = MSEcriterion(z_pred, z_true.detach())
            else:
                idx = compute_alignment(z_pred.data.cpu(), z_true.data.cpu(), alignment = args.alignment)
                z_true_aligned = torch.cat([z_true[i,idx[i],:].unsqueeze(0) for i in range(len(idx))], dim=0)
                errE_L2 = MSEcriterion(z_pred, z_true_aligned.detach())

        errE = errE_L1
        if not args.no_EL2:
            errE = errE + errE_L2
        if not args.no_EL3:
            errE = errE + errE_L3

        tot_errE += errE.item()
        ne += 1
        errE.backward()
        optimizerE.step()

        iters += 1

    # Check how the generator is doing by saving G's output on fixed_noise
    if  epoch % args.print_freq == 0:
        N = float(len(dataloader))
        val_error, miou = test.validation_error(valloader, netG, netE)
        ng = max(ng, 1); nd = max(nd, 1); ne = max(1,ne)
        print('%s:[%d/%s][%d] D %.3f: G %.3f: E %.3f: val=%.3f bval=%.3f miou=%.3f: biou=%.3f: D(x): %.3f: D(G(z)): %.3f / %.3f'
                  % (args.obj_name, epoch, args.seed, iters, tot_errD/nd, tot_errG/ng, tot_errE/ne,
                     val_error, best_val_error, miou, best_miou, D_x/nd, D_G_z1/nd, D_G_z2/ng))

        if epoch == 0 or epoch == 1: print('time-taken=%0.4f'%(time.time()-tt))

        if tot_errG/ng < 10.0 and tot_errD/nd < 10.0:
            output = netG(fixed_input)
            input_outputs = torch.cat([input_samples[:16], output[:16]], dim=0)
            vutils.save_image(input_outputs.detach().cpu().mul(0.5).add(0.5),
                              '{0}/inputs_and_generated_samples-{1}-{2}.png'
                              .format(args.experiment, args.seed, args.obj_name), padding=2, nrow=16, pad_value=255)
            vutils.save_image(input_outputs.detach().cpu().mul(0.5).add(0.5),
                              '{0}/all/inputs_and_generated_samples/{1}.png'
                              .format(args.experiment, epoch), padding=2, nrow=16, pad_value=255)

            if val_error < best_val_error:
                vutils.save_image(input_outputs.detach().cpu().mul(0.5).add(0.5),
                                  '{0}/best_inputs_and_generated_samples-{1}-{2}.png'
                                  .format(args.experiment, args.seed, args.obj_name), padding=2, nrow=16, pad_value=255)

            output = netG(fixed_input[0].unsqueeze(0))
            input_outputs = torch.cat([input_samples[:16], output[:16]], dim=0)
            vutils.save_image(input_outputs.detach().cpu().mul(0.5).add(0.5),
                              '{0}/single_generated_samples-{1}-{2}.png'
                              .format(args.experiment, args.seed, args.obj_name), padding=2, nrow=16,pad_value=255)
            vutils.save_image(input_outputs.detach().cpu().mul(0.5).add(0.5),
                              '{0}/all/single_generated_samples/{1}.png'
                              .format(args.experiment, epoch), padding=2, nrow=16,pad_value=255)

            if val_error < best_val_error:
                vutils.save_image(input_outputs.detach().cpu().mul(0.5).add(0.5),
                                  '{0}/best_single_generated_samples-{1}-{2}.png'
                                  .format(args.experiment, args.seed, args.obj_name), padding=2, nrow=16,pad_value=255)

            inputs, outputs, single_insts = test.encoder_decoder(input_samples[:16], netG, netE)

            input_outputs = torch.cat([inputs[:16], outputs[:16]], dim=0)
            segments = test.find_instance_segments(single_insts, args.depth_threshold)
            input_outputs = torch.cat([input_outputs.repeat(1,3,1,1).detach().cpu().mul(0.5).add(0.5), segments/255], dim=0)

            vutils.save_image(input_outputs, '{0}/inputs_and_decoded_inputs-{1}-{2}.png'
                              .format(args.experiment, args.seed, args.obj_name), padding=2,nrow=16, pad_value=255)

            if miou >= best_miou:
                input_outputs = torch.cat([input_outputs, torch.cat(single_insts, dim=0).repeat(1,3,1,1).detach().cpu().mul(0.5).add(0.5)], dim=0)
                np.save('{0}/best_inputs_and_decoded_inputs-{1}-{2}.npy'
                        .format(args.experiment, args.seed, args.obj_name), input_outputs.numpy())
                vutils.save_image(input_outputs, '{0}/best_inputs_and_decoded_inputs-{1}-{2}.png'
                                  .format(args.experiment, args.seed, args.obj_name), padding=2,nrow=16, pad_value=255)

            if val_error < best_val_error:
                save_model(netD, errD.data.item(), epoch, 'netD', args.experiment, args.trail_id)
                save_model(netG, errG.data.item(), epoch, 'netG', args.experiment, args.trail_id)
                save_model(netE, errE.data.item(), epoch, 'netE', args.experiment, args.trail_id)

    # if the models are diverging or going unstable, lets reset the models from a previous good epoch.
    if tot_errG/ng >40.0 or tot_errD/nd > 40.0:
            print('resetting the training as things are (G, D, E) are diverging!: updading G trial_id=%d'%(args.trail_id))
            print('-----------------------------------------------')
            try:
                netG.load_state_dict(torch.load(os.path.join(args.experiment, 'ckpt_netG.pth'))['net'])
            except:
                netG, _, _, _, _, _ = init_models(netG, netD, netE)
            optimizerG = optim.Adam(list(netG.dec1.parameters())+list(netG.dec2.parameters()), lr=args.lr, betas=(args.beta1, 0.999))

    if val_error <= best_val_error:
        best_val_error = val_error
    if miou > best_miou:
        best_miou = miou
