from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class train_total(nn.Module):
    def __init__(self, model1, model2):
        super(train_total, self).__init__()
        self.popular = model1
        self.unpopular = model2
        self.criterion = nn.TripletMarginLoss(margin=1)

    def forward(self, x, bg_popular, bg_unpopular, beta, average=True):

        loss_pop, RE_pop, KL_pop = self.popular.calculate_loss(x, bg_popular, beta, average=True, cl_criterion=self.criterion)
        loss_unpop, RE_unpop, KL_unpop = self.unpopular.calculate_loss(x, bg_unpopular, beta, average=True, cl_criterion=self.criterion)

        loss = (loss_pop + loss_unpop) / 2 
        RE = (RE_pop + RE_unpop) / 2
        KL = (KL_pop + KL_unpop) / 2

        return loss, RE, KL


def train_vae(epoch, args, train_loader, popular_loader, unpopular_loader, model, optimizer):

    train_loss = 0
    train_re = 0
    train_kl = 0

    model.train()

    if args.warmup == 0:
        beta = args.max_beta
    else:
        beta = args.max_beta * epoch / args.warmup
        if beta > args.max_beta:
            beta = args.max_beta
    print('beta: {}'.format(beta))

    bg_popular_iterator = iter(popular_loader)
    bg_unpopular_iterator = iter(unpopular_loader)

    for batch_idx, (data, target) in enumerate(train_loader):    
        try:
            bg_popular, _ = next(bg_popular_iterator)
            bg_unpopular, _ = next(bg_unpopular_iterator)
        except StopIteration:
            bg_popular_iterator = iter(popular_loader)
            bg_popular, _ = next(bg_popular_iterator)
            bg_unpopular_iterator = iter(unpopular_loader)
            bg_unpopular, _ = next(bg_unpopular_iterator)

        if args.cuda:
            data, target, bg_popular, bg_unpopular = data.cuda(), target.cuda(), bg_popular.cuda(), bg_unpopular.cuda()

        data, target, bg_popular, bg_unpopular = data.float(), target.float(), bg_popular.float(), bg_unpopular.float()
        data, target, bg_popular, bg_unpopular = Variable(data), Variable(target), Variable(bg_popular), Variable(bg_unpopular) 
        
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        optimizer.zero_grad()
        loss, RE, KL =  model(x, bg_popular, bg_unpopular, beta, average=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        train_re += -RE.data.item()
        train_kl += KL.data.item()

    # calculate final loss
    train_loss /= len(train_loader)
    train_re /= len(train_loader)
    train_kl /= len(train_loader)

    return model, train_loss, train_re, train_kl

