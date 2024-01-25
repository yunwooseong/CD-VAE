from __future__ import print_function

import numpy as np

import math

from scipy.special import logsumexp

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable
from torch.nn.functional import normalize

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256, log_Softmax
from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# H+Vamp model with 2 or more hidden layers use this model
#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        self.args = args

        # encoder: q(z | x)
        modules = [nn.Dropout(p=0.5),
                   NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.q_z_layers = nn.Sequential(*modules)

        self.q_z_mean = Linear(self.args.hidden_size, self.args.z_size)
        self.q_z_logvar = NonLinear(self.args.hidden_size, self.args.z_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # encoder: q(s2 | x)
        modules = [nn.Dropout(p=0.5),
                   NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.q_s2_layers = nn.Sequential(*modules)

        self.q_s2_mean = Linear(self.args.hidden_size, self.args.s1_size)
        self.q_s2_logvar = NonLinear(self.args.hidden_size, self.args.s1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # encoder: q(s1 | x, s2)
        self.q_s1_layers_x = nn.Sequential(
            nn.Dropout(p=0.5),
            NonLinear(np.prod(self.args.input_size), self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )
        self.q_s1_layers_s2 = nn.Sequential(
            NonLinear(self.args.s2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )

        modules = [NonLinear(2 * self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()),]
        for _ in range(0, self.args.num_layers - 2):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.q_s1_layers_joint = nn.Sequential(*modules)

        self.q_s1_mean = Linear(self.args.hidden_size, self.args.s1_size)
        self.q_s1_logvar = NonLinear(self.args.hidden_size, self.args.s1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # decoder: p(s1 | s2)
        modules = [NonLinear(self.args.s2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()),]
        for _ in range(0, self.args.num_layers - 1):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.p_s1_layers = nn.Sequential(*modules)

        self.p_s1_mean = Linear(self.args.hidden_size, self.args.s1_size)
        self.p_s1_logvar = NonLinear(self.args.hidden_size, self.args.s1_size, activation=nn.Hardtanh(min_val=-12.,max_val=4.))

        # decoder: p(x | s1, s2, z)
        self.p_x_layers_s1 = nn.Sequential(
            NonLinear(self.args.s1_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )
        self.p_x_layers_s2 = nn.Sequential(
            NonLinear(self.args.s2_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )
        self.p_x_layers_z = nn.Sequential(
            NonLinear(self.args.z_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())
        )
        modules = [NonLinear(3 * self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh())]
        for _ in range(0, self.args.num_layers - 2):
            modules.append(NonLinear(self.args.hidden_size, self.args.hidden_size, gated=self.args.gated, activation=nn.Tanh()))
        self.p_x_layers_jointall = nn.Sequential(*modules)

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        if self.args.input_type == 'multinomial':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=None)
        elif self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))    

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        # add pseudo-inputs for VampPrior
        self.add_pseudoinputs()

    # AUXILIARY METHODS
    def calculate_loss(self, trainloader, bgloader, beta=1., average=False, cl_criterion=None):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_mean, z_logvar, tg_z, tg_z_mu, tg_z_logvar, bg_z, bg_z_mu, bg_z_logvar, tg_s1, tg_s1_mu, tg_s1_logvar, tg_s2, tg_s2_mu, tg_s2_logvar, s1_p_mean, s1_p_logvar = self.forward(trainloader, bgloader) #,tg_s

        # TG_RE
        if self.args.input_type == 'binary':
            TG_RE = log_Bernoulli(trainloader, x_mean, dim=1)
        elif self.args.input_type == 'multinomial':
            TG_RE = log_Softmax(trainloader, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            TG_RE = -log_Logistic_256(trainloader, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')
        
        # BG_RE
        if self.args.input_type == 'binary':
            BG_RE = log_Bernoulli(bgloader, z_mean, dim=1)
        elif self.args.input_type == 'multinomial':
            BG_RE = log_Softmax(bgloader, z_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            BG_RE = -log_Logistic_256(bgloader, z_mean, z_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')
        
        RE = torch.mean((TG_RE + BG_RE))

        # KL
        log_q_bg_z = log_Normal_diag(bg_z, bg_z_mu, bg_z_logvar, dim=1)
        log_p_bg_z = self.log_p_z(bg_z)
        log_q_tg_z = log_Normal_diag(tg_z, tg_z_mu, tg_z_logvar, dim=1)
        log_p_tg_z = self.log_p_z(tg_z)
        log_p_s1 = log_Normal_diag(tg_s1, s1_p_mean, s1_p_logvar, dim=1)
        log_q_s1 = log_Normal_diag(tg_s1, tg_s1_mu, tg_s1_logvar, dim=1)
        log_p_s2 = self.log_p_s2(tg_s2)
        log_q_s2 = log_Normal_diag(tg_s2, tg_s2_mu, tg_s2_logvar, dim=1)
        KL = - (log_p_s1 + log_p_s2 + log_p_bg_z + log_p_tg_z - log_q_s1 - log_q_s2 - log_q_bg_z - log_q_tg_z)

        triplet_loss = cl_criterion(bg_z, tg_z, tg_s1)
        
        loss = -RE + beta * KL + triplet_loss

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    # THE MODEL: VARIATIONAL POSTERIOR
    def q_z(self, x):
        x = self.q_z_layers(x)

        z_q_mean = self.q_z_mean(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mean, z_q_logvar

    def q_s2(self, x):
        x = self.q_s2_layers(x)

        s2_q_mean = self.q_s2_mean(x)
        s2_q_logvar = self.q_s2_logvar(x)
        return s2_q_mean, s2_q_logvar

    def q_s1(self, x, s2):
        x = self.q_s1_layers_x(x)

        s2 = self.q_s1_layers_s2(s2)

        h = torch.cat((x,s2), dim=-1)

        h = self.q_s1_layers_joint(h)

        s1_q_mean = self.q_s1_mean(h)
        s1_q_logvar = self.q_s1_logvar(h)
        return s1_q_mean, s1_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_s1(self, s2):
        s2 = self.p_s1_layers(s2)

        s1_mean = self.p_s1_mean(s2)
        s1_logvar = self.p_s1_logvar(s2)
        return s1_mean, s1_logvar
    
    def p_x(self, s1, s2, z):
        s1 = self.p_x_layers_s1(s1)

        s2 = self.p_x_layers_s2(s2)

        z = self.p_x_layers_z(z)

        h = torch.cat((s1, s2 ,z), dim=-1)
    
        h = self.p_x_layers_jointall(h)

        x_mean = self.p_x_mean(h)
        if self.args.input_type == 'binary' or self.args.input_type == 'multinomial':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.p_x_logvar(h)
        return x_mean, x_logvar

    # the prior
    def log_p_s2(self, s2):
        # vamp prior
        # z2 - MB x M
        C = self.args.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        s2_p_mean, s2_p_logvar = self.q_s2(X)  # C x M

        # expand z
        z_expand = s2.unsqueeze(1)
        means = s2_p_mean.unsqueeze(0)
        logvars = s2_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB
        # calculte log-sum-exp
        log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1)))  # MB

        return log_prior
    
    def log_p_z(self, z):
        # vamp prior
        # z - MB x M
        C = self.args.number_components

        # calculate params
        X = self.means(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.q_z(X)  # C x M

        # expand z
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        return log_prior
    
    # THE MODEL: FORWARD PASS
    def forward(self, trainloader, bgloader):
        # input normalization
        trainloader = normalize(trainloader, dim=1)
        bgloader = normalize(bgloader, dim=1)

        # z ~ q(z | x)
        tg_z_mu, tg_z_logvar = self.q_z(trainloader)
        tg_z = self.reparameterize(tg_z_mu, tg_z_logvar)

        bg_z_mu, bg_z_logvar = self.q_z(bgloader)
        bg_z = self.reparameterize(bg_z_mu, bg_z_logvar)
        
        # s2 ~ q(s2 | x)
        tg_s2_mu, tg_s2_logvar = self.q_s2(trainloader)
        tg_s2 = self.reparameterize(tg_s2_mu, tg_s2_logvar)

        # s1 ~ q(s1 | x, s2)
        tg_s1_mu, tg_s1_logvar = self.q_s1(trainloader, tg_s2)
        tg_s1 = self.reparameterize(tg_s1_mu, tg_s1_logvar)

        # p(s1 | s2)
        s1_p_mean, s1_p_logvar = self.p_s1(tg_s2)
        
        # x_mean = p(x|s1,s2)
        x_mean, x_logvar = self.p_x(tg_s1, tg_s2, tg_z)
        
        # x_mean = p(x|s1,s2)
        zeros = torch.zeros_like(bg_z)
        z_mean, z_logvar = self.p_x(zeros, zeros, bg_z)

        return x_mean, x_logvar, z_mean, z_logvar, tg_z, tg_z_mu, tg_z_logvar, bg_z, bg_z_mu, bg_z_logvar, tg_s1, tg_s1_mu, tg_s1_logvar, tg_s2, tg_s2_mu, tg_s2_logvar, s1_p_mean, s1_p_logvar #,tg_s
