
from torchreid import metrics
from torchreid.utils import re_ranking
import numpy as np
import torch

from torch.nn import functional as F

if __name__ == '__main__':
    ff = torch.load('train_avg_feature.pt')
    f_pids = np.load('train_pids.npy')
    f_camids = np.load('train_camids.npy')
    

    # eval_index = 50
    # ef = ff[eval_index].unsqueeze(0).clone()
    # e_pids = f_pids[eval_index].reshape((1,))
    # e_camids = f_camids[eval_index].reshape((1,))

    ef = ff.clone()
    dist_metric='euclidean'
    print(
        'Computing distance matrix with metric={} ...'.format(dist_metric)
    )
    # normalize feature
    # print('Normalizing feature ...')
    # qf = F.normalize(qf, p=2, dim=1)
    # gf = F.normalize(gf, p=2, dim=1)

    distmat = metrics.compute_distance_matrix(ff, ef, dist_metric)
    distmat = distmat.numpy()

    # print('Applying person re-ranking ...')
    # distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
    # distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
    # distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        f_pids,
        f_pids,
        f_camids,
        f_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    ef = torch.zeros_like(ff)


    distmat = metrics.compute_distance_matrix(ff, ef, dist_metric)
    distmat = distmat.numpy()

    # print('Applying person re-ranking ...')
    # distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
    # distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
    # distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    print('Computing CMC and mAP ...')
    zcmc, zmAP = metrics.evaluate_rank(
        distmat,
        f_pids,
        f_pids,
        f_camids,
        f_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(zmAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, zcmc[r - 1]))
    print(zmAP - mAP)
    print(sum(zcmc[:20] - cmc[:20])+ (zmAP - mAP))
    
