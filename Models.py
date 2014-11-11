from joblib   import Parallel, delayed
from settings import *



def update_model(model):
  model.train()

def run_model(model):
  return model.validate()

class Models:
  def __init__(self, models):
    self.models = models

  def train(self):
    pool= Parallel(n_jobs = num_cores)
    pool(delayed(update_model)(model) for model in self.models)

  def validation(self):
    pool = Parallel(n_jobs = num_cores)
    pool(delayed(run_model)(model) for model in self.models)
