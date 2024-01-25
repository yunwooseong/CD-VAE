from __future__ import print_function

import torch
import torch.utils.data as data_utils

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz
import os
import random
import pickle
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def set_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    #torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# ======================================================================================================================
def load_ml20m(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("", "ML_20m", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    
    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te
    
    def load_bg_data_popular():
        print("Making background dataset for Movielens-20m . . .")
        
        train_data_csr = load_train_data(os.path.join("", "ML_20m", 'train.csv'))
        coo = coo_matrix(train_data_csr)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        train_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense() 

        item_popularity = torch.sum(train_tensor, dim=0, dtype=int)
        torch.set_printoptions(sci_mode=False)
        pop_distribution = torch.Tensor(item_popularity / torch.sum(item_popularity))

        sorted_pop_distribution, sorted_indices = torch.sort(pop_distribution, descending=True)

        top_10_percent_threshold = torch.quantile(sorted_pop_distribution, q=0.9)
        top_10_percent_items = (pop_distribution > top_10_percent_threshold).nonzero().flatten()

        data_bg = torch.zeros_like(train_tensor)
        sampling_count = round(len(pop_distribution) / 10)

        set_seed(args.seed)
        for i in range(train_tensor.shape[0]):
            out = torch.multinomial(pop_distribution, sampling_count, replacement=False)
            popular_items_from_sampled = [index for index in out if index in top_10_percent_items]
            popular_items = torch.tensor(popular_items_from_sampled)
            popular_items = popular_items.long()
            data_bg[i][popular_items] = 1

        bg_train = sparse.csr_matrix((data_bg[data_bg!=0], 
                                      torch.nonzero(data_bg, as_tuple=True)), dtype='float64',
                                     shape=train_data_csr.shape)

        print("Done!")
        return bg_train

    def load_bg_data_unpopular():
        print("Making background dataset for Movielens-20m . . .")

        train_data_csr = load_train_data(os.path.join("", "ML_20m", 'train.csv'))
        coo = coo_matrix(train_data_csr)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        train_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        item_popularity = torch.sum(train_tensor, dim=0, dtype=int)
        torch.set_printoptions(sci_mode=False)
        pop_distribution = torch.Tensor(item_popularity / torch.sum(item_popularity))

        sorted_pop_distribution, sorted_indices = torch.sort(pop_distribution, descending=True)

        max_popularity = torch.max(pop_distribution)
        reversed_pop_distribution = max_popularity - pop_distribution

        bottom_10_percent_threshold = torch.quantile(sorted_pop_distribution, q=0.1)
        bottom_10_percent_items = (pop_distribution <= bottom_10_percent_threshold).nonzero().flatten()

        data_bg = torch.zeros_like(train_tensor)
        sampling_count = round(len(pop_distribution) / 10)

        set_seed(args.seed)
        for i in range(train_tensor.shape[0]):
            out = torch.multinomial(reversed_pop_distribution, sampling_count, replacement=False)
            unpopular_items_from_sampled = [index for index in out if index in bottom_10_percent_items]
            unpopular_items = torch.tensor(unpopular_items_from_sampled)
            unpopular_items = unpopular_items.long()
            data_bg[i][unpopular_items] = 1

        bg_train = sparse.csr_matrix((data_bg[data_bg != 0],
                                      torch.nonzero(data_bg, as_tuple=True)), dtype='float64',
                                     shape=train_data_csr.shape)

        print("Done!")
        return bg_train

    # train, validation and test data
    tg_train = load_train_data(os.path.join("", "ML_20m", 'train.csv'))
    np.random.shuffle(tg_train)
    bg_train = load_bg_data_popular()
    #bg_train = load_bg_data_unpopular()

    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join("", "ML_20m", 'validation_tr.csv'),
                                         os.path.join("", "ML_20m", 'validation_te.csv'))

    test_data_tr, test_data_te = load_tr_te_data(os.path.join("", "ML_20m", 'test_tr.csv'),
                                           os.path.join("", "ML_20m", 'test_te.csv'))

    # idle y's
    y_t_train = np.zeros((tg_train.shape[0], 1))
    y_b_train = np.zeros((bg_train.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(tg_train), torch.from_numpy(y_t_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    background = data_utils.TensorDataset(torch.from_numpy(bg_train.toarray()), torch.from_numpy(y_b_train))
    bgloader = data_utils.DataLoader(background, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(vad_data_tr), torch.from_numpy(vad_data_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(test_data_tr).float(), torch.from_numpy(test_data_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
    
        # For tg_train encoder initialization
        init_train = tg_train[0:args.number_components].T
        args.pseudoinputs_mean_tg = torch.from_numpy(init_train + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components)).float()

    return train_loader, bgloader, val_loader, test_loader, args

# ======================================================================================================================
def load_netflix(args, **kwargs):

    unique_sid = list()
    with open(os.path.join("", "Netflix", 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    # set args
    args.input_size = [1, 1, n_items]
    if args.input_type != "multinomial":
        args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float32',
                                 shape=(n_users, n_items)).toarray()
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        data_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)),
                                    dtype='float32', shape=(end_idx - start_idx + 1, n_items)).toarray()
        return data_tr, data_te

    def load_bg_data_popular():
        print("Making background dataset for Netflix . . .")

        train_data_csr = load_train_data(os.path.join("", "Netflix", 'train.csv'))
        coo = coo_matrix(train_data_csr)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        train_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        item_popularity = torch.sum(train_tensor, dim=0, dtype=int)
        torch.set_printoptions(sci_mode=False)
        pop_distribution = torch.Tensor(item_popularity / torch.sum(item_popularity))

        sorted_pop_distribution, sorted_indices = torch.sort(pop_distribution, descending=True)

        top_10_percent_threshold = torch.quantile(sorted_pop_distribution, q=0.9)
        top_10_percent_items = (pop_distribution > top_10_percent_threshold).nonzero().flatten()

        data_bg = torch.zeros_like(train_tensor)
        sampling_count = round(len(pop_distribution) / 10)

        set_seed(args.seed)
        for i in range(train_tensor.shape[0]):
            out = torch.multinomial(pop_distribution, sampling_count, replacement=False)
            popular_items_from_sampled = [index for index in out if index in top_10_percent_items]
            popular_items = torch.tensor(popular_items_from_sampled)
            popular_items = popular_items.long()
            data_bg[i][popular_items] = 1

        bg_train = sparse.csr_matrix((data_bg[data_bg != 0],
                                      torch.nonzero(data_bg, as_tuple=True)), dtype='float64',
                                     shape=train_data_csr.shape)

        print("Done!")
        return bg_train

    def load_bg_data_unpopular():
        print("Making background dataset for Netflix . . .")

        train_data_csr = load_train_data(os.path.join("", "Netflix", 'train.csv'))
        coo = coo_matrix(train_data_csr)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        train_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        item_popularity = torch.sum(train_tensor, dim=0, dtype=int)
        torch.set_printoptions(sci_mode=False)
        pop_distribution = torch.Tensor(item_popularity / torch.sum(item_popularity))

        sorted_pop_distribution, sorted_indices = torch.sort(pop_distribution, descending=True)

        max_popularity = torch.max(pop_distribution)
        reversed_pop_distribution = max_popularity - pop_distribution

        bottom_10_percent_threshold = torch.quantile(sorted_pop_distribution, q=0.1)
        bottom_10_percent_items = (pop_distribution <= bottom_10_percent_threshold).nonzero().flatten()

        data_bg = torch.zeros_like(train_tensor)
        sampling_count  = round(len(pop_distribution) / 10)

        set_seed(args.seed)
        for i in range(train_tensor.shape[0]):
            out = torch.multinomial(reversed_pop_distribution, sampling_count, replacement=False)
            unpopular_items_from_sampled = [index for index in out if index in bottom_10_percent_items]
            unpopular_items = torch.tensor(unpopular_items_from_sampled)
            unpopular_items = unpopular_items.long()
            data_bg[i][unpopular_items] = 1

        bg_train = sparse.csr_matrix((data_bg[data_bg != 0],
                                      torch.nonzero(data_bg, as_tuple=True)), dtype='float64',
                                     shape=train_data_csr.shape)

        print("Done!")

        return bg_train

    # train, validation and test data
    tg_train = load_train_data(os.path.join("", "pro_sg", 'train.csv'))
    np.random.shuffle(tg_train)
    bg_train = load_bg_data_popular()
    #bg_train = load_bg_data_unpopular()
    
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join("", "Netflix", 'validation_tr.csv'),
                                         os.path.join("", "Netflix", 'validation_te.csv'))

    test_data_tr, test_data_te = load_tr_te_data(os.path.join("", "Netflix", 'test_tr.csv'),
                                           os.path.join("", "Netflix", 'test_te.csv'))

    # idle y's
    y_t_train = np.zeros((tg_train.shape[0], 1))
    y_b_train = np.zeros((bg_train.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(tg_train), torch.from_numpy(y_t_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    background = data_utils.TensorDataset(torch.from_numpy(bg_train.toarray()), torch.from_numpy(y_b_train))
    bgloader = data_utils.DataLoader(background, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(vad_data_tr), torch.from_numpy(vad_data_te))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(test_data_tr).float(), torch.from_numpy(test_data_te))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.use_training_data_init == 1:
        args.pseudoinputs_std = 0.01
    
        # For tg_train encoder initialization
        init_train = tg_train[0:args.number_components].T
        args.pseudoinputs_mean_tg = torch.from_numpy(init_train + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components)).float()

    return train_loader, bgloader, val_loader, test_loader, args


# ======================================================================================================================
def load_dataset(args, **kwargs):
    if args.dataset_name == 'ml20m':
        train_loader, bgloader, val_loader, test_loader, args = load_ml20m(args, **kwargs)
    elif args.dataset_name == 'netflix':
        train_loader, bgloader, val_loader, test_loader, args = load_netflix(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')
    
    return train_loader, bgloader, val_loader, test_loader, args

