#!/usr/bin/env python

from multiprocessing  import Pool
from settings         import *
from multiprocessing  import Process, Queue
from ModelExemple     import *

class Models:
  def __init__(self, models):
    self.models = models
    self.para = True

  def train(self):
    if self.para:
      pool = Pool(processes=num_cores)
      self.models = pool.map(lambda x : x.train(),self.models)
      pool.close()
    else:
      for model in self.models :
        model.train()

  def validation(self):
    if self.para:
      pool = Pool(processes=num_cores)
      self.models = pool.map(lambda x : x.validate(),self.models)
      pool.close()
    else:
      for model in self.models :
        model.validate()

  def dump(self):
    for model in self.models :
      model.dump_score()

