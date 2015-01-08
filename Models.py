#!/usr/bin/env python

from settings         import *
from multiprocessing  import Pool, Process, Queue
from ModelExemple     import *

class Models:
  def __init__(self, models,online=False):
    if online:
      self.gen_params = models #Format : [(modelClass, modelKwargs)]
    else:
      self.models = models
    self.para = False
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
    if self.para:
      raise Exception("Not implemented yet")
    else:
      if self.online:
        models = self.gen_models()
      else:
        models = self.models
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


