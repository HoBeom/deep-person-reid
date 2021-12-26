
from torchreid import metrics
from torchreid.utils import re_ranking
import numpy as np
import torch

from torch.nn import functional as F

if __name__ == '__main__':
    qf = torch.load('query_avg_feature.pt')
    q_pids = np.load('query_pids.npy')
    q_camids = np.load('query_camids.npy')
    gf = torch.load('gallery_avg_feature.pt')
    g_pids = np.load('gallery_pids.npy')
    g_camids = np.load('gallery_camids.npy')


    dist_metric='euclidean'
    print(
        'Computing distance matrix with metric={} ...'.format(dist_metric)
    )
    # normalize feature
    # print('Normalizing feature ...')
    # qf = F.normalize(qf, p=2, dim=1)
    # gf = F.normalize(gf, p=2, dim=1)

    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    # print('Applying person re-ranking ...')
    # distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
    # distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
    # distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
