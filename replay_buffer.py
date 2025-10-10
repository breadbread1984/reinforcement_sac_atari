#!/usr/bin/python3

import random
import pickle
import zlib
import lmdb

class ReplayBuffer(object):
  def __init__(self, file_path = 'replay_buffer.lmdb'):
    self.db = lmdb.open(file_path, map_size = 1099511627776)
    self.latest_id = 0
  def add(self, sample):
    with self.db.begin(write = True) as txn:
      txn.put(str(self.latest_id).zfill(5).encode(), zlib.compress(pickle.dumps(sample)))
    self.latest_id += 1
  def keys(self):
    with self.db.begin() as txn:
      keys = [key for key, _ in txn.cursor()]
    return keys
  def size(self):
    keys = self.keys()
    return len(keys)
  def sample(self, sample_num):
    keys = self.keys()
    sampled = random.sample(keys, sample_num)
    dataset = list()
    with self.db.begin() as txn:
      for key in sampled:
        data = txn.get(key)
        sample = pickle.loads(zlib.decompress(data))
        dataset.append(sample)
    return dataset
  def truncate(self, size = 10000):
    if self.size() <= size: return
    keys = list(sorted(self.keys()))
    to_delete = keys[:-size]
    with self.db.begin(write = True) as txn:
      for key in to_delete:
        txn.delete(key)
