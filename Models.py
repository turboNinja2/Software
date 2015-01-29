#!/usr/bin/env python

from settings         import *
from multiprocessing  import Pool, Process, Queue
from ModelExemple     import *

class Models:
  def __init__(self, models,online=False,para=False):
    if online:
      self.gen_params = models #Format : [(modelClass, modelKwargs)]
    else:
      self.models = models
    self.para = para
    self.online = online
  
  def gen_models(self):
    for model, kwargs in self.gen_params:
      yield model(**kwargs)
    
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

  def train_validated_dump_and_clear(self):
    result = []
    if self.online:
      models = self.gen_models()
    else:
      models = self.models
    if self.para:
      def f(m):
        m.train()
        m.validate()
        m.dump_score()
        r = m.params, m.getValidationLogLoss()
        del m
        return r

      pool = Pool(processes=num_cores)
      result = pool.map(lambda x : f(x),models)
      pool.close()
    else:
      for model in models :
        model.train()
        model.validate()
        model.dump_score()
        result.append((model.params, model.getValidationLogLoss()))
        del model
    return result
  
  def writeSubmissions(self):
    for model in self.models :
      model.writeSubmission()


