#!/usr/bin/python3

import random
import pickle
import zlib
import lmdb
from uuid import uuid4

class ReplayBuffer(object):
  def __init__(self, file_path = 'replay_buffer.lmdb'):
    self.db = lmdb.open(file_path, map_size = 1099511627776)
  def add(self, sample):
    with self.db.begin(write = True) as txn:
      txn.put(str(uuid4()).encode(), zlib.compress(pickle.dumps(sample)))
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
  def truncate(self, size = 500000):
    if self.size() <= size: return
    keys = random.shuffle(self.keys())
    to_delete = keys[:-size]
    with self.db.begin(write = True) as txn:
      for key in to_delete:
        txn.delete(key)
