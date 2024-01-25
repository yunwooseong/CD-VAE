from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def train_vae(epoch, args, train_loader, bgloader, model, optimizer):
    cl_criterion = nn.TripletMarginLoss(margin=1)
    
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0

    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = args.max_beta
    else:
        beta = args.max_beta * epoch / args.warmup
        if beta > args.max_beta:
            beta = args.max_beta
    print('beta: {}'.format(beta))
    
    bg_iterator = iter(bgloader)

    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            bg_data, _ = next(bg_iterator)
        except StopIteration:
            bg_iterator = iter(bgloader)
            bg_data, _ = next(bg_iterator)

        if args.cuda:
            data, target, bg_data = data.cuda(), target.cuda(), bg_data.cuda()

        data, target, bg_data = data.float(), target.float(), bg_data.float()
        data, target, bg_data = Variable(data), Variable(target), Variable(bg_data)
        
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data
        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.calculate_loss(x, bg_data, beta, average=True, cl_criterion=cl_criterion)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.data.item()
        train_re += -RE.data.item()
        train_kl += KL.data.item()

    # calculate final loss
    train_loss /= len(train_loader)
    train_re /= len(train_loader) 
    train_kl /= len(train_loader) 

    return model, train_loss, train_re, train_kl