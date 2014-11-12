from joblib           import Parallel, delayed
from multiprocessing  import Pool
from settings         import *

def update_model(model):
  model.train()
  return model

def run_model(model):
  model.validate()
  return model

def dump_model(model):
  model.dump_score()
  return model


class Models:
  def __init__(self, models):
    self.models = models
    self.para = True

  def train(self):
    if self.para:
      pool = Parallel(n_jobs = num_cores)
      self.models = pool(delayed(update_model)(model) for model in self.models)
    else:
      for model in self.models :
        model.train()

  def validation(self):
    if self.para:
      pool = Parallel(n_jobs = num_cores)
      self.models = pool(delayed(run_model)(model) for model in self.models)
    else:
      for model in self.models :
        model.validate()

  def dump(self):
    if self.para:
      pool = Parallel(n_jobs = num_cores)
      self.models = pool(delayed(dump_model)(model) for model in self.models)
    else:
      for model in self.models :
        model.dump_score()
