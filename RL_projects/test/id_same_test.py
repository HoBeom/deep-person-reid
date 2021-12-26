import numpy as np

a = np.load('all_avg_feature/train_pids.npy')
b = np.load('../reid_gym/frist_16_feature/train_pids.npy')
print(a.shape)
print(b.shape)
print((a==b).all())