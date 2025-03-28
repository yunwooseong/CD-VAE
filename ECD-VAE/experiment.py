from __future__ import print_function
import argparse

import torch
import torch.optim as optim

from utils.optimizer import AdamNormGrad

import os
import numpy as np
import datetime

from utils.load_data_both import load_dataset
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description='VAE+VampPrior')
# arguments for optimization
parser.add_argument('--batch_size', type=int, default=200, metavar='BStrain',
                    help='input batch size for training (default: 200)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='BStest',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=400, metavar='E',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                    help='learning rate (default: 0.00025)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warm-up')
parser.add_argument('--max_beta', type=float, default=1., metavar='B',
                    help='maximum value of beta for training')

# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

# model: latent size, input_size, so on
parser.add_argument('--num_layers', type=int, default=2, metavar='NL',
                    help='number of layers')

parser.add_argument('--s1_size', type=int, default=200, metavar='MS',
                    help='latent size of s1')
parser.add_argument('--s2_size', type=int, default=200, metavar='MS',
                    help='latent size of s2')
parser.add_argument('--z_size', type=int, default=200, metavar='MZ',
                    help='latent size of z')
parser.add_argument('--hidden_size', type=int, default=600, metavar="H",
                    help='the width of hidden layers')
parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--number_components', type=int, default=1000, metavar='NC',
                    help='number of pseudo-inputs')
parser.add_argument('--pseudoinputs_mean', type=float, default=0.05, metavar='PM',
                    help='mean for init pseudo-inputs')
parser.add_argument('--pseudoinputs_std', type=float, default=0.01, metavar='PS',
                    help='std for init pseudo-inputs')

parser.add_argument('--use_training_data_init', action='store_true', default=False,
                    help='initialize pseudo-inputs with randomly chosen training data')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous, multinomial')

parser.add_argument('--gated', action='store_true', default=False,
                    help='use gating mechanism')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='ml20m', metavar='DN',
                    help='name of the dataset:  ml20m, netflix')

parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

# note
parser.add_argument('--note', type=str, default="none", metavar='NT',
                    help='additional note on the experiment')
parser.add_argument('--no_log', action='store_true', default=False,
                    help='print log to log_dir')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}  #! Changed num_workers: 1->0 because of error

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:10]

    model_name = args.dataset_name + '_' + \
                 '(K_' + str(args.number_components) + ')' + \
                 '_' + args.input_type + '_beta(' + str(args.max_beta) + ')' + \
                 '_layers(' + str(args.num_layers) + ')' + '_hidden(' + str(args.hidden_size) + ')' + \
                 '_s1(' + str(args.s1_size) + ')' + '_s2(' + str(args.s2_size) + ')' + '_z(' + str(args.z_size) + ')'

    # DIRECTORY FOR SAVING
    snapshots_path = ''
    dir = snapshots_path + args.model_signature + '_' + model_name + '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    train_loader, popular_loader, unpopular_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    from models.HVamp import VAE

    model1 = VAE(args)
    if args.cuda:
        model1.cuda()

    model2 = VAE(args)
    if args.cuda:
        model2.cuda()

    # ======================================================================================================================
    print(args)
    log_dir = "vae_experiment_log_" + str(os.getenv("COMPUTERNAME")) +".txt"

    open(log_dir, 'a').close()

    # ======================================================================================================================
    print('perform experiment')
    from utils.perform_experiment import experiment_vae
    experiment_vae(args, train_loader, popular_loader, unpopular_loader, val_loader, test_loader, model1, model2, dir, log_dir, model_name = 'Hvamp')
    # ======================================================================================================================


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
