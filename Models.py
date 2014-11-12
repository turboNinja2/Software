from joblib           import Parallel, delayed
from multiprocessing  import *
from settings         import *

def update_model(model):
  model.train()

def run_model(model):
  return model.validate()

def dump_model(model):
  model.dump_score()

class Models:
  def __init__(self, models):
    self.models = models

  def train(self):
    pool = Parallel(n_jobs = num_cores)
    pool(delayed(update_model)(model) for model in self.models)

  def validation(self):
    pool = Parallel(n_jobs = num_cores)
    pool(delayed(run_model)(model) for model in self.models)

  def dump(self):
    pool = Parallel(n_jobs = num_cores)
    pool(delayed(dump_model)(model) for model in self.models)
