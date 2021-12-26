import torch
import numpy as np
import torchreid
from torchreid.utils import FeatureExtractor
from glob import glob
from tqdm import tqdm

img_train = glob('./data/mars/bbox_train/*/*.jpg')
img_test = glob('./data/mars/bbox_test/*/*.jpg')
img_train.sort()
img_test.sort()
print(len(img_train), len(img_test))
print(img_train[0])
print(img_test[0])

extractor = FeatureExtractor(
model_name='osnet_x1_0',
model_path='./data/trained_model/osnet_x1_0_market_trained.pth.tar-250',
device='cuda'
)

for img_path in tqdm(img_train):
    feature_path = img_path.replace('jpg', 'npy')
    feature = extractor(img_path)
    np.save(feature_path, feature.cpu().numpy())

for img_path in tqdm(img_test):
    feature_path = img_path.replace('jpg', 'npy')
    feature = extractor(img_path)
    np.save(feature_path, feature.cpu().numpy())

