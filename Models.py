#!/usr/bin/env python


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
    for model in self.models :
      model.dump_score()



def script():
  
  #PARAMS
  from test import Test
  test = Test()
  params  = test.params
  model   = test.Learning
  kwargs  = test.kwargs

  step = 5 
  step_range = 8

  #FIRST SCALE
  init_scale = lambda x : pow(10,-x)
  model_list = []
  for i in xrange(step_range):
    model_list.append(model({"alpha":init_scale(i)},test.w,**test.kwargs))

  models = Models(model_list)
  models.train()
  models.validation()

  
  x1,x2 = truc(models)

  #ITERATIONS
  for i in xrange(step):
    scale = lambda x : x2 + float(x)*(x1-x2)/(step_range-1)
    model_list = []
    for j in xrange(step_range):
      model_list.append(model({"alpha":scale(j)},test.w,**test.kwargs))
    models = Models(model_list)
    models.train()
    models.validation()
    x1,x2 = truc(models)

def truc(models):
  r = map(lambda x : (x.score,x.params["alpha"]),models.models)
  print r
  r_min = 10
  i_min = -1
  for i,value in enumerate(r):
    if value[0] < r_min:
      r_min = value[0]
      i_min = i
  if i_min == 0:
    x1 = r[0][1]
    x2 = r[1][1]
  elif i_min == len(r)-1:
    x1 = r[len(r)-1][1]
    x2 = r[len(r)][1]
  else:
    x1 = r[i_min-1][1]
    x2 = r[i_min+1][1]
    
  print x1
  print x2
  return x1,x2



script()
