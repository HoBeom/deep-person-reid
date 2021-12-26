from collections import defaultdict
import pickle
import gym
import torch

import numpy as np
from torchreid import metrics
from torchreid.utils import re_ranking
from gym import spaces
from random import randint
from torch.nn import functional as F


import logging

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('reid_gym.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ReidVideoV1(gym.Env):
  """Video Reid Environment that follows gym interface"""
  # possible actions
  N_DISCRETE_ACTIONS = 3
  PASS = 0
  MIX = 1 # Hierarchical mixing ? -> v2
  END = 2

  def __init__(self, cfg):
    super(ReidVideoV1, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.test_mode = cfg['TEST_MODE']
    self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=10, shape=(1027,))
    # Example for using image as input:
    # Example when using discrete actions:
    self.train_all_avg_feature = torch.load("frist_16_feature/train_all_avg_feature.pt")
    self.train_feature = torch.load("frist_16_feature/train_16_all_feature.pt")
    self.avg_feature = torch.load("frist_16_feature/train_16_avg_feature.pt")
    self.max_feature_idx = self.train_feature.size(0)-1
    self.max_position = self.train_feature.size(1)-1
    self.reset = self._train_reset
    self.step = self._train_step
    if self.test_mode:
      self.dist_metric = 'euclidean'
      self.query_all_avg_feature = torch.load("frist_16_feature/query_all_avg_feature.pt")
      self.gallery_all_avg_feature = torch.load("frist_16_feature/gallery_all_avg_feature.pt")
      self.query_feature = torch.load("frist_16_feature/query_16_all_feature.pt")
      self.query_avg_feature = torch.load("frist_16_feature/query_16_avg_feature.pt")
      self.gallery_feature = torch.load("frist_16_feature/gallery_16_all_feature.pt")
      self.gallery_avg_feature = torch.load("frist_16_feature/gallery_16_avg_feature.pt")
      self.q_pids = np.load("frist_16_feature/query_pids.npy")
      self.g_pids = np.load("frist_16_feature/gallery_pids.npy")
      self.q_camids = np.load("frist_16_feature/query_camids.npy")
      self.g_camids = np.load("frist_16_feature/gallery_camids.npy")
      self.query_num = self.query_feature.size(0)
      self.train_all_avg_feature = torch.cat((self.query_all_avg_feature, self.gallery_all_avg_feature), dim=0)
      self.train_feature = torch.cat((self.query_feature, self.gallery_feature), dim=0)
      self.avg_feature = torch.cat((self.query_avg_feature, self.gallery_avg_feature), dim=0)
      self.max_feature_idx = self.train_feature.size(0)-1
      self.max_position = self.train_feature.size(1)-1
      self.eval_feautre = torch.zeros_like(self.train_all_avg_feature)
      self.position_count = np.zeros(self.train_all_avg_feature.size(0))
      self.mix_count = np.zeros(self.train_all_avg_feature.size(0))
      self.reset = self._test_reset
      self.step = self._test_step
    self.reset()

    
  def get_next_person_idx(self):
    return randint(0, self.max_feature_idx)

  def mix_feature(self, feature_A, feature_B, mix_rate=0.5):
    return feature_A * mix_rate + feature_B * (1 - mix_rate)

  def create_observation(self, feature_A, mixed_feature):
    diff_mean = torch.abs(torch.mean(feature_A - mixed_feature)).unsqueeze(0)
    diff_max = torch.abs(torch.max(feature_A - mixed_feature)).unsqueeze(0)
    diff_min = torch.abs(torch.min(feature_A - mixed_feature)).unsqueeze(0)
    return torch.cat((feature_A, mixed_feature,  diff_max, diff_min, diff_mean), dim=0).numpy()

  def get_pre_observation(self):
    return torch.tensor(self.pre_observation)

  def get_cur_person_feature(self):
    return self.train_feature[self.cur_feature_idx][self.position]

  def get_avg_feature(self):
    return self.avg_feature[self.cur_feature_idx]

  def get_distance(self, person_feature):
    # Computes euclidean squared distance.
    avg_feature = self.train_all_avg_feature[self.cur_feature_idx]
    avg_feature = torch.pow(avg_feature, 2)
    person_feature = torch.pow(person_feature, 2)
    return torch.sum(avg_feature - person_feature).item()

  def _train_step(self, action):
    # action: 0: pass, 1: mix, 2: end
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        reward = -reward
      elif action == self.MIX:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        done = True
    else:
      if action == self.MIX:
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        # reward = self.pre_distance - self.get_distance(origin_avg_feature)
        reward = -30
        done = True
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = {'pre_distance': self.pre_distance, 'cur_distance': cur_distance, 'reward': reward}
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position:
      done = True
    return observation, reward, done, info

  def _test_step(self, action):
    # TODO test mode
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
        reward = -reward
      elif action == self.MIX:
        self.position += 1
        self.mix_count[self.cur_feature_idx] += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = cur_mixed_feature
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        done = True
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
    else:
      if action == self.MIX:
        self.mix_count[self.cur_feature_idx] += 1
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        reward = -100
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = None
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position or done:
      if self.cur_feature_idx % 1000 == 0:
        logging.info(f'feature_idx: {self.cur_feature_idx}/{self.max_feature_idx}')
      self.position_count[self.cur_feature_idx] = self.position
      done = False
      self.init_mix_flag = False
      if self.cur_feature_idx == self.max_feature_idx:
        done = True
        self.eval()
      # save result
      info = {}
      info['cur_feature_idx'] = self.cur_feature_idx
      info['position'] = self.position
      info['num_features'] = self.max_feature_idx
      self.position = 0
      self.cur_feature_idx += 1
    return observation, reward, done, info

  def _train_reset(self):
    self.cur_feature_idx = self.get_next_person_idx()
    self.position = 0
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def _test_reset(self):
    self.cur_feature_idx = 0
    self.position = 0
    self.position_count = np.zeros(self.train_all_avg_feature.size(0))
    self.mix_count = np.zeros(self.train_all_avg_feature.size(0))
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def render(self, mode='human'):
    pass

  def close (self):
    pass
  
  def eval(self, normalize=False, rerank=False, save_path=None):
    # split data query, gallery
    qf = self.eval_feautre[:self.query_num]
    gf = self.eval_feautre[self.query_num:]
    dist_metric = self.dist_metric
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if normalize:
      print('Normalizing feature ...')
      qf = F.normalize(qf, p=2, dim=1)
      gf = F.normalize(gf, p=2, dim=1)

    if rerank:
      print('Applying person re-ranking ...')
      distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
      distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
      distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    
    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        self.q_pids,
        self.g_pids,
        self.q_camids,
        self.g_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    if save_path is not None:
      print('Saving results to {}'.format(save_path))
      if save_path.endswith('.pt'):
        torch.save(self.eval_feautre, save_path)
      elif save_path.endswith('.npy'):
        np.save(save_path, self.eval_feautre.numpy())
      else:
        raise ValueError('Unsupported file format: {}'.format(save_path))
    avg_use_image = np.average(self.position_count)
    avg_mix_image = np.average(self.mix_count)
    print(f'use/view images: {avg_mix_image}/{avg_use_image}')
    return cmc, mAP, (avg_mix_image, avg_use_image)

class ReidVideoV2(gym.Env):
  """Video Reid Environment that follows gym interface"""
  # possible actions
  N_DISCRETE_ACTIONS = 3
  PASS = 0
  MIX = 1 # Hierarchical mixing !!
  END = 2
  MIX_FEATURE_SIZE = 512


  def __init__(self, cfg):
    super(ReidVideoV2, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.test_mode = cfg['TEST_MODE']
    # self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
    # TODO test mix rate action
    self.action_space = spaces.Tuple((
      spaces.Discrete(self.N_DISCRETE_ACTIONS),
      spaces.Box(low=0, high=1, shape=(self.MIX_FEATURE_SIZE,), dtype=np.float32),
    ))
    self.observation_space = spaces.Box(low=0, high=10, shape=(1024,))
    # Example for using image as input:
    # Example when using discrete actions:
    self.train_all_avg_feature = torch.load("frist_16_feature/train_all_avg_feature.pt")
    self.train_feature = torch.load("frist_16_feature/train_16_all_feature.pt")
    self.avg_feature = torch.load("frist_16_feature/train_16_avg_feature.pt")
    self.max_feature_idx = self.train_feature.size(0)-1
    self.max_position = self.train_feature.size(1)-1
    self.reset = self._train_reset
    self.step = self._train_step
    if self.test_mode:
      self.dist_metric = 'euclidean'
      self.query_all_avg_feature = torch.load("frist_16_feature/query_all_avg_feature.pt")
      self.gallery_all_avg_feature = torch.load("frist_16_feature/gallery_all_avg_feature.pt")
      self.query_feature = torch.load("frist_16_feature/query_16_all_feature.pt")
      self.query_avg_feature = torch.load("frist_16_feature/query_16_avg_feature.pt")
      self.gallery_feature = torch.load("frist_16_feature/gallery_16_all_feature.pt")
      self.gallery_avg_feature = torch.load("frist_16_feature/gallery_16_avg_feature.pt")
      self.q_pids = np.load("frist_16_feature/query_pids.npy")
      self.g_pids = np.load("frist_16_feature/gallery_pids.npy")
      self.q_camids = np.load("frist_16_feature/query_camids.npy")
      self.g_camids = np.load("frist_16_feature/gallery_camids.npy")
      self.query_num = self.query_feature.size(0)
      self.train_all_avg_feature = torch.cat((self.query_all_avg_feature, self.gallery_all_avg_feature), dim=0)
      self.train_feature = torch.cat((self.query_feature, self.gallery_feature), dim=0)
      self.avg_feature = torch.cat((self.query_avg_feature, self.gallery_avg_feature), dim=0)
      self.max_feature_idx = self.train_feature.size(0)-1
      self.max_position = self.train_feature.size(1)-1
      self.eval_feautre = torch.zeros_like(self.train_all_avg_feature)
      self.reset = self._test_reset
      self.step = self._test_step
    self.reset()

    
  def get_next_person_idx(self):
    return randint(0, self.max_feature_idx)

  def mix_feature(self, feature_A, feature_B, mix_rate=0.5):
    return feature_A * mix_rate + feature_B * (1 - mix_rate)

  def create_observation(self, feature_A, mixed_feature):
    return torch.cat((feature_A, mixed_feature), dim=0).numpy()

  def get_pre_observation(self):
    return torch.tensor(self.pre_observation)

  def get_cur_person_feature(self):
    return self.train_feature[self.cur_feature_idx][self.position]

  def get_avg_feature(self):
    return self.avg_feature[self.cur_feature_idx]

  def get_distance(self, person_feature):
    # Computes euclidean squared distance.
    avg_feature = self.train_all_avg_feature[self.cur_feature_idx]
    avg_feature = torch.pow(avg_feature, 2)
    person_feature = torch.pow(person_feature, 2)
    return torch.sum(avg_feature - person_feature).item()

  def _train_step(self, action):
    # action: (action, mix_rate)
    action, mix_rate = action
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature, mix_rate)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        reward = -reward
      elif action == self.MIX:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        reward += 3
        done = True
    else:
      if action == self.MIX:
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        reward = self.pre_distance - self.get_distance(person_feature)
        reward -= 3
        done = True
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = {'pre_distance': self.pre_distance, 'cur_distance': cur_distance, 'reward': reward}
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position:
      done = True
    return observation, reward, done, info

  def _test_step(self, action):
    # TODO test mode
    action, mix_rate = action
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature, mix_rate)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
        reward = -reward
      elif action == self.MIX:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = cur_mixed_feature
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        reward += 3
        done = True
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
    else:
      if action == self.MIX:
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        done = True
        reward = self.pre_distance - self.get_distance(person_feature)
        reward -= 3
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = None
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position or done:
      done = False
      self.init_mix_flag = False
      if self.cur_feature_idx == self.max_feature_idx:
        done = True
        self.eval()
      # save result
      info = {}
      info['cur_feature_idx'] = self.cur_feature_idx
      info['position'] = self.position
      info['num_features'] = self.max_feature_idx
      self.position = 0
      self.cur_feature_idx += 1
    return observation, reward, done, info

  def _train_reset(self):
    self.cur_feature_idx = self.get_next_person_idx()
    self.position = 0
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def _test_reset(self):
    self.cur_feature_idx = 0
    self.position = 0
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def render(self, mode='human'):
    pass

  def close (self):
    pass
  
  def eval(self, normalize=False, rerank=False, save_path=None):
    # split data query, gallery
    qf = self.eval_feautre[:self.query_num]
    gf = self.eval_feautre[self.query_num:]
    dist_metric = self.dist_metric
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if normalize:
      print('Normalizing feature ...')
      qf = F.normalize(qf, p=2, dim=1)
      gf = F.normalize(gf, p=2, dim=1)

    if rerank:
      print('Applying person re-ranking ...')
      distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
      distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
      distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    
    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        self.q_pids,
        self.g_pids,
        self.q_camids,
        self.g_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    if save_path is not None:
      print('Saving results to {}'.format(save_path))
      if save_path.endswith('.pt'):
        torch.save(self.eval_feautre, save_path)
      elif save_path.endswith('.npy'):
        np.save(save_path, self.eval_feautre.numpy())
      else:
        raise ValueError('Unsupported file format: {}'.format(save_path))
    return cmc, mAP


class ReidVideoV3(gym.Env):
  """Video Reid Environment that follows gym interface"""
  # possible actions
  N_DISCRETE_ACTIONS = 2
  PASS = 0
  MIX = 1
  # END = 2, # NO END

  def __init__(self, cfg):
    super(ReidVideoV3, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.test_mode = cfg['TEST_MODE']
    self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=10, shape=(1027,))
    # Example for using image as input:
    # Example when using discrete actions:
    self.train_all_avg_feature = torch.load("frist_16_feature/train_all_avg_feature.pt")
    self.train_feature = torch.load("frist_16_feature/train_16_all_feature.pt")
    self.avg_feature = torch.load("frist_16_feature/train_16_avg_feature.pt")
    self.max_feature_idx = self.train_feature.size(0)-1
    self.max_position = self.train_feature.size(1)-1
    self.reset = self._train_reset
    self.step = self._train_step
    if self.test_mode:
      self.dist_metric = 'euclidean'
      self.query_all_avg_feature = torch.load("frist_16_feature/query_all_avg_feature.pt")
      self.gallery_all_avg_feature = torch.load("frist_16_feature/gallery_all_avg_feature.pt")
      self.query_feature = torch.load("frist_16_feature/query_16_all_feature.pt")
      self.query_avg_feature = torch.load("frist_16_feature/query_16_avg_feature.pt")
      self.gallery_feature = torch.load("frist_16_feature/gallery_16_all_feature.pt")
      self.gallery_avg_feature = torch.load("frist_16_feature/gallery_16_avg_feature.pt")
      self.q_pids = np.load("frist_16_feature/query_pids.npy")
      self.g_pids = np.load("frist_16_feature/gallery_pids.npy")
      self.q_camids = np.load("frist_16_feature/query_camids.npy")
      self.g_camids = np.load("frist_16_feature/gallery_camids.npy")
      self.query_num = self.query_feature.size(0)
      self.train_all_avg_feature = torch.cat((self.query_all_avg_feature, self.gallery_all_avg_feature), dim=0)
      self.train_feature = torch.cat((self.query_feature, self.gallery_feature), dim=0)
      self.avg_feature = torch.cat((self.query_avg_feature, self.gallery_avg_feature), dim=0)
      self.max_feature_idx = self.train_feature.size(0)-1
      self.max_position = self.train_feature.size(1)-1
      self.eval_feautre = torch.zeros_like(self.train_all_avg_feature)
      self.position_count = np.zeros(self.train_all_avg_feature.size(0))
      self.mix_count = np.zeros(self.train_all_avg_feature.size(0))
      self.reset = self._test_reset
      self.step = self._test_step
    self.reset()

    
  def get_next_person_idx(self):
    return randint(0, self.max_feature_idx)

  def mix_feature(self, feature_A, feature_B, mix_rate=0.5):
    return feature_A * mix_rate + feature_B * (1 - mix_rate)

  def create_observation(self, feature_A, mixed_feature):
    diff_mean = torch.abs(torch.mean(feature_A - mixed_feature)).unsqueeze(0)
    diff_max = torch.abs(torch.max(feature_A - mixed_feature)).unsqueeze(0)
    diff_min = torch.abs(torch.min(feature_A - mixed_feature)).unsqueeze(0)
    return torch.cat((feature_A, mixed_feature,  diff_max, diff_min, diff_mean), dim=0).numpy()

  def get_pre_observation(self):
    return torch.tensor(self.pre_observation)

  def get_cur_person_feature(self):
    return self.train_feature[self.cur_feature_idx][self.position]

  def get_avg_feature(self):
    return self.avg_feature[self.cur_feature_idx]

  def get_distance(self, person_feature):
    # Computes euclidean squared distance.
    avg_feature = self.train_all_avg_feature[self.cur_feature_idx]
    avg_feature = torch.pow(avg_feature, 2)
    person_feature = torch.pow(person_feature, 2)
    return torch.sum(avg_feature - person_feature).item()

  def _train_step(self, action):
    # action: 0: pass, 1: mix, 2: end
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        reward = -reward

      elif action == self.MIX:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
      # elif action == self.END:
      #   observation = self.pre_observation
      #   origin_avg_feature = self.get_avg_feature()
      #   reward = self.pre_distance - self.get_distance(origin_avg_feature)
      #   done = True
    else:
      if action == self.MIX:
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
      elif self.position == self.max_position: #and action == self.END:
        observation = self.pre_observation
        reward = -30
        done = True
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = {'pre_distance': self.pre_distance, 'cur_distance': cur_distance, 'reward': reward}
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position:
      done = True
    return observation, reward, done, info

  def _test_step(self, action):
    # TODO test mode
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
        reward = -reward
      elif action == self.MIX:
        self.position += 1
        self.mix_count[self.cur_feature_idx] += 1        
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = cur_mixed_feature
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        done = True
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
    else:
      if action == self.MIX:
        self.init_mix_flag = True
        self.position += 1
        self.mix_count[self.cur_feature_idx] += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        reward = -100
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = None
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position or done:
      done = False
      self.position_count[self.cur_feature_idx] = self.position
      self.init_mix_flag = False
      if self.cur_feature_idx == self.max_feature_idx:
        done = True
        self.eval()
      # save result
      info = {}
      info['cur_feature_idx'] = self.cur_feature_idx
      info['position'] = self.position
      info['num_features'] = self.max_feature_idx
      self.position = 0
      self.cur_feature_idx += 1
    return observation, reward, done, info

  def _train_reset(self):
    self.cur_feature_idx = self.get_next_person_idx()
    self.position = 0
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def _test_reset(self):
    self.cur_feature_idx = 0
    self.position = 0
    self.position_count = np.zeros(self.train_all_avg_feature.size(0))
    self.mix_count = np.zeros(self.train_all_avg_feature.size(0))
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def render(self, mode='human'):
    pass

  def close (self):
    pass
  
  def eval(self, normalize=False, rerank=False, save_path=None):
    # split data query, gallery
    qf = self.eval_feautre[:self.query_num]
    gf = self.eval_feautre[self.query_num:]
    dist_metric = self.dist_metric
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if normalize:
      print('Normalizing feature ...')
      qf = F.normalize(qf, p=2, dim=1)
      gf = F.normalize(gf, p=2, dim=1)

    if rerank:
      print('Applying person re-ranking ...')
      distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
      distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
      distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    
    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        self.q_pids,
        self.g_pids,
        self.q_camids,
        self.g_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    if save_path is not None:
      print('Saving results to {}'.format(save_path))
      if save_path.endswith('.pt'):
        torch.save(self.eval_feautre, save_path)
      elif save_path.endswith('.npy'):
        np.save(save_path, self.eval_feautre.numpy())
      else:
        raise ValueError('Unsupported file format: {}'.format(save_path))
    avg_use_image = np.average(self.position_count)
    avg_mix_image = np.average(self.mix_count)
    print(f'use/view images: {avg_mix_image}/{avg_use_image}')
    return cmc, mAP, (avg_mix_image, avg_use_image)


class ReidVideoV4(gym.Env):
  """Video Reid Environment that follows gym interface"""
  # possible actions
  N_DISCRETE_ACTIONS = 3
  PASS = 0
  MIX = 1 
  END = 2

  def __init__(self, cfg):
    super(ReidVideoV4, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.test_mode = cfg['TEST_MODE']
    self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
    self.observation_space = spaces.Box(low=0, high=10, shape=(1027,))
    # Example for using image as input:
    # Example when using discrete actions:
    self.train_all_avg_feature = torch.load("frist_16_feature/train_all_avg_feature.pt")
    self.train_feature = torch.load("frist_16_feature/train_16_all_feature.pt")
    self.avg_feature = torch.load("frist_16_feature/train_16_avg_feature.pt")
    self.t_pids = np.load("frist_16_feature/train_pids.npy")
    self.t_camids = np.load("frist_16_feature/train_camids.npy")
    self.id2idx = defaultdict(list)
    self.idx2id = dict()
    for idx, pid in enumerate(self.t_camids):
      self.id2idx[pid].append(idx)
      self.idx2id[idx] = pid
    for ids in self.id2idx.values():
      self.train_all_avg_feature[ids] = torch.mean(self.train_all_avg_feature[ids], dim=0)    

    self.max_feature_idx = self.train_feature.size(0)-1
    self.max_position = self.train_feature.size(1)-1
    self.reset = self._train_reset
    self.step = self._train_step
    if self.test_mode:
      self.dist_metric = 'euclidean'
      self.query_all_avg_feature = torch.load("frist_16_feature/query_all_avg_feature.pt")
      self.gallery_all_avg_feature = torch.load("frist_16_feature/gallery_all_avg_feature.pt")
      self.query_feature = torch.load("frist_16_feature/query_16_all_feature.pt")
      self.query_avg_feature = torch.load("frist_16_feature/query_16_avg_feature.pt")
      self.gallery_feature = torch.load("frist_16_feature/gallery_16_all_feature.pt")
      self.gallery_avg_feature = torch.load("frist_16_feature/gallery_16_avg_feature.pt")
      self.q_pids = np.load("frist_16_feature/query_pids.npy")
      self.g_pids = np.load("frist_16_feature/gallery_pids.npy")
      self.q_camids = np.load("frist_16_feature/query_camids.npy")
      self.g_camids = np.load("frist_16_feature/gallery_camids.npy")
      self.query_num = self.query_feature.size(0)
      self.train_all_avg_feature = torch.cat((self.query_all_avg_feature, self.gallery_all_avg_feature), dim=0)
      self.train_feature = torch.cat((self.query_feature, self.gallery_feature), dim=0)
      self.avg_feature = torch.cat((self.query_avg_feature, self.gallery_avg_feature), dim=0)
      self.max_feature_idx = self.train_feature.size(0)-1
      self.max_position = self.train_feature.size(1)-1
      self.eval_feautre = torch.zeros_like(self.train_all_avg_feature)
      self.position_count = np.zeros(self.train_all_avg_feature.size(0))
      self.mix_count = np.zeros(self.train_all_avg_feature.size(0))
      self.reset = self._test_reset
      self.step = self._test_step
    self.reset()

    
  def get_next_person_idx(self):
    return randint(0, self.max_feature_idx)

  def mix_feature(self, feature_A, feature_B, mix_rate=0.5):
    return feature_A * mix_rate + feature_B * (1 - mix_rate)

  def create_observation(self, feature_A, mixed_feature):
    diff_mean = torch.abs(torch.mean(feature_A - mixed_feature)).unsqueeze(0)
    diff_max = torch.abs(torch.max(feature_A - mixed_feature)).unsqueeze(0)
    diff_min = torch.abs(torch.min(feature_A - mixed_feature)).unsqueeze(0)
    return torch.cat((feature_A, mixed_feature,  diff_max, diff_min, diff_mean), dim=0).numpy()

  def get_pre_observation(self):
    return torch.tensor(self.pre_observation)

  def get_cur_person_feature(self):
    return self.train_feature[self.cur_feature_idx][self.position]

  def get_avg_feature(self):
    return self.avg_feature[self.cur_feature_idx]

  def get_distance(self, person_feature):
    # Computes euclidean squared distance.
    avg_feature = self.train_all_avg_feature[self.cur_feature_idx]
    # avg_feature = torch.pow(avg_feature, 2)
    # person_feature = torch.pow(person_feature, 2)
    return torch.sum(avg_feature - person_feature).item()

  def _train_step(self, action):
    # action: 0: pass, 1: mix, 2: end
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        # reward = -reward
        reward = 0
      elif action == self.MIX:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
        reward = 0
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        # reward += 10
        done = True
    else:
      if action == self.MIX:
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
        reward = 0
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        reward = -30
        done = True
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        # reward = -reward
        reward = 0

    info = {'pre_distance': self.pre_distance, 'cur_distance': cur_distance, 'reward': reward}
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position:
      done = True
    return observation, reward, done, info

  def _test_step(self, action):
    # TODO test mode
    person_feature = self.get_cur_person_feature()
    observation = self.pre_observation.copy()
    pre_mixed_feature = self.get_pre_observation()[512:1024]
    cur_mixed_feature = self.mix_feature(pre_mixed_feature, person_feature)
    cur_distance = self.get_distance(cur_mixed_feature)
    reward = self.pre_distance - cur_distance
    done = False
    if self.init_mix_flag:
      if action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, pre_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
        reward = -reward
      elif action == self.MIX:
        self.position += 1
        self.mix_count[self.cur_feature_idx] += 1
        person_feature = self.get_cur_person_feature()
        observation = self.create_observation(person_feature, cur_mixed_feature)
        self.eval_feautre[self.cur_feature_idx] = cur_mixed_feature
      elif action == self.END:
        observation = self.pre_observation
        origin_avg_feature = self.get_avg_feature()
        reward = self.pre_distance - self.get_distance(origin_avg_feature)
        done = True
        self.eval_feautre[self.cur_feature_idx] = pre_mixed_feature
    else:
      if action == self.MIX:
        self.mix_count[self.cur_feature_idx] += 1
        self.init_mix_flag = True
        self.position += 1
        next_person_feature = self.get_cur_person_feature()
        observation = self.create_observation(next_person_feature, person_feature)
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif self.position == self.max_position and action == self.END:
        observation = self.pre_observation
        reward = -100
        self.eval_feautre[self.cur_feature_idx] = person_feature
      elif action == self.PASS:
        self.position += 1
        person_feature = self.get_cur_person_feature()
        mixed_feature = person_feature.clone()
        observation = self.create_observation(person_feature, mixed_feature)
        reward = -reward

    info = None
    self.pre_observation = observation
    self.pre_distance = cur_distance
    if self.position == self.max_position or done:
      if self.cur_feature_idx % 1000 == 0:
        logging.info(f'feature_idx: {self.cur_feature_idx}/{self.max_feature_idx}')
      self.position_count[self.cur_feature_idx] = self.position
      done = False
      self.init_mix_flag = False
      if self.cur_feature_idx == self.max_feature_idx:
        done = True
        self.eval()
      # save result
      info = {}
      info['cur_feature_idx'] = self.cur_feature_idx
      info['position'] = self.position
      info['num_features'] = self.max_feature_idx
      self.position = 0
      self.cur_feature_idx += 1
    return observation, reward, done, info

  def _train_reset(self):
    self.cur_feature_idx = self.get_next_person_idx()
    self.position = 0
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def _test_reset(self):
    self.cur_feature_idx = 0
    self.position = 0
    self.position_count = np.zeros(self.train_all_avg_feature.size(0))
    self.mix_count = np.zeros(self.train_all_avg_feature.size(0))
    person_feature = self.get_cur_person_feature()
    mixed_feature = person_feature.clone()
    observation = self.create_observation(person_feature, mixed_feature)
    self.pre_observation = observation.copy()
    self.pre_distance = self.get_distance(person_feature)
    self.init_mix_flag = False
    return observation

  def render(self, mode='human'):
    pass

  def close (self):
    pass
  
  def eval(self, normalize=False, rerank=False, save_path=None):
    # split data query, gallery
    qf = self.eval_feautre[:self.query_num]
    gf = self.eval_feautre[self.query_num:]
    dist_metric = self.dist_metric
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if normalize:
      print('Normalizing feature ...')
      qf = F.normalize(qf, p=2, dim=1)
      gf = F.normalize(gf, p=2, dim=1)

    if rerank:
      print('Applying person re-ranking ...')
      distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
      distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
      distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    
    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        self.q_pids,
        self.g_pids,
        self.q_camids,
        self.g_camids,
    )
    ranks=[1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    if save_path is not None:
      print('Saving results to {}'.format(save_path))
      if save_path.endswith('.pt'):
        torch.save(self.eval_feautre, save_path)
      elif save_path.endswith('.npy'):
        np.save(save_path, self.eval_feautre.numpy())
      else:
        raise ValueError('Unsupported file format: {}'.format(save_path))
    avg_use_image = np.average(self.position_count)
    avg_mix_image = np.average(self.mix_count)
    print(f'use/view images: {avg_mix_image}/{avg_use_image}')
    return cmc, mAP, (avg_mix_image, avg_use_image)


if __name__ == '__main__':
  from ray.rllib.agents.dqn import DQNTrainer
  # test create env
  # ReidVideoV1({"TEST_MODE": True})
  # ReidVideoV1({"TEST_MODE": False})
  
  # Create an RLlib Trainer instance.
  env = ReidVideoV1({"TEST_MODE": True})
  trainer = DQNTrainer(
      config={
          "env": ReidVideoV1,
          "env_config": {
              "TEST_MODE": False
          },
          "num_workers": 4,
          "framework": "torch",
          "num_gpus": 1,
          "model": {
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "relu",
          },
      })
  
  def eval_logging(env, iter):
    ranks = [1, 5, 10, 20]
    cmc, mAP, num_images = env.eval(save_path="eval_result.pt")
    logger.info(f'** Use {num_images} image per video **')
    logger.info(f'** Iter {iter} Results **')
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('CMC curve')
    for r in ranks:
        logger.info('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    
    cmc, mAP, num_images = env.eval(normalize=True, save_path="eval_N_result.pt")
    logger.info(f'** Iter {iter} normalize Results **')
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('CMC curve')
    for r in ranks:
        logger.info('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    
    cmc, mAP, num_images = env.eval(rerank=True, save_path="eval_R_result.pt")
    logger.info(f'** Iter {iter} Rerank Results **')
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('CMC curve')
    for r in ranks:
        logger.info('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    
    cmc, mAP, num_images = env.eval(normalize=True, rerank=True, save_path="eval_NR_result.pt")
    logger.info(f'** Iter {iter} Rerank Results **')
    logger.info('mAP: {:.1%}'.format(mAP))
    logger.info('CMC curve')
    for r in ranks:
        logger.info('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

  for i in range(5):
    eval_logging(env, i)
    for j in range(200):
        results = trainer.train()
        logger.info(f"Iter: {i}:{j}; avg. reward={results['episode_reward_mean']}")
    
    obs = env.reset()
    done = False
    while not done:
        action = trainer.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
  eval_logging(env, 20)
