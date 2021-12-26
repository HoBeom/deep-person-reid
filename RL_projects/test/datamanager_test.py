from math import pi
from re import I
import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn
import numpy as np

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)
from test_default_config import videodata_kwargs, get_default_config
from tqdm import tqdm

def parse_data_for_eval(data):
    features = data['feature']
    pids = data['pid']
    camids = data['camid']
    return features, pids, camids

def feature_loader(data_loader, load_type='avg'):
    if load_type == 'avg':
        load_type = torch.mean
    else:
        load_type = lambda x, y: x
    data_lenth = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    sqeuence_len = data_loader.dataset.seq_len
    
    f_ = np.zeros(shape=(data_lenth, 512), dtype=np.float32)
    pids_ = np.zeros(shape=(data_lenth), dtype=np.int64)
    camids_ = np.zeros(shape=(data_lenth), dtype=np.int64)
    all_ = np.zeros(shape=(data_lenth, sqeuence_len, 512), dtype=np.float32)
    # all_ = []
    idxs = [i for i in range(batch_size)]
    for batch_idx, data in tqdm(enumerate(data_loader)):
        batch_idxs = [(batch_idx-1) * batch_size + i for i in range(batch_size)]
        features, pids, camids = parse_data_for_eval(data)
        f_[batch_idxs] = load_type(features, dim=1).numpy()[idxs]
        pids_[batch_idxs] = pids[idxs]
        camids_[batch_idxs] = camids[idxs]
        all_[batch_idxs] = features[idxs]

    f_ = torch.tensor(f_, dtype=torch.float32)
    all_ = torch.tensor(all_, dtype=torch.float32)
    # pids_ = np.asarray(pids_)
    # camids_ = np.asarray(camids_)
    return f_, pids_, camids_, all_

if __name__ == '__main__':
    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    cfg.merge_from_file('test_config.yaml')
    set_random_seed(cfg.train.seed)

    datamanager = torchreid.data.VideoRLDataManager(**videodata_kwargs(cfg))

    train_loader = datamanager.train_loader
    query_loader = datamanager.test_loader['mars']['query']
    gallery_loader = datamanager.test_loader['mars']['gallery']

    print('Extracting features from train set ...')
    tf, t_pids, t_camids, t_all = feature_loader(train_loader)
    print('Done, obtained {}-by-{} matrix'.format(tf.size(0), tf.size(1)))
    torch.save(tf, 'train_avg_feature.pt')
    torch.save(t_all, 'train_all_feature.pt')
    np.save('train_pids.npy', t_pids)
    np.save('train_camids.npy', t_camids)

    print('Extracting features from query set ...')
    qf, q_pids, q_camids, q_all = feature_loader(query_loader)
    print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))
    torch.save(qf, './query_avg_feature.pt')
    torch.save(q_all, './query_all_feature.pt')
    np.save('./query_pids.npy', q_pids)
    np.save('./query_camids.npy', q_camids)

    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids, g_all = feature_loader(gallery_loader)
    print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))
    torch.save(gf, './gallery_avg_feature.pt')
    torch.save(g_all, './gallery_all_feature.pt')
    np.save('./gallery_pids.npy', g_pids)
    np.save('./gallery_camids.npy', g_camids)