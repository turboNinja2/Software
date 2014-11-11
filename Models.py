from joblib   import Parallel, delayed
from settings import *

def update_model(model, path):
  return model.run(path)

def run_model(model,path):
  return model.run(path,update=False)

class Models:
  def __init__(self, models):
    self.models = models

  def train(self, trainPath):
    pool= Parallel(n_jobs = num_cores)
    pool(delayed(update_model)(model,trainPath) for model in self.models)

  def validation(self, validationPath):
    pool = Parallel(n_jobs = num_cores)
    pool(delayed(run_model)(model,validationPath) for model in self.models)
