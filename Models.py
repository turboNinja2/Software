#!/usr/bin/env python


from joblib           import Parallel, delayed
from multiprocessing  import Pool
from settings         import *
from multiprocessing  import Process, Queue
from ModelExemple     import *

def tem_update(model):
  model.train()
  return model

def update_model(q,model):
  q.put(tem_update(model))

def tem_run(model):
  model.validate()
  return model

def run_model(q,model):
  q.put(tem_run(model))


class Models:
  def __init__(self, models):
    self.models = models
    self.para = True
    self.q = Queue()

  def train(self):
    if self.para:
      ps = map(lambda x : Process(target=update_model, args=(self.q,x)), self.models)
      for p in ps:
        p.start()
      models = []
      for i in xrange(len(ps)):
        models.append(self.q.get())
      self.models = models
    else:
      for model in self.models :
        model.train()

  def validation(self):
    if self.para:
      ps = map(lambda x : Process(target=run_model, args=(self.q,x)), self.models)
      for p in ps:
        p.start()
      models = []
      for i in xrange(len(ps)):
        models.append(self.q.get())
      self.models = models
    else:
      for model in self.models :
        model.validate()

  def dump(self):
    for model in self.models :
      model.dump_score()



class SoftwareTM():
  
  def __init__(self):
    self.step       = 3 
    self.step_range = 5

  def compute_x12(self):
    self.models.train()
    self.models.validation()
    x1,x2 = truc(self.models)
    return x1, x2

  def build_model_list(self, scale_function):
    model_list = []
    for i in xrange(self.step_range):
      model_list.append(mediumModel(scale_function(i)))
    return model_list


  def first_scale(self):
    init_scale = lambda x : pow(10,-x)
    model_list = self.build_model_list(init_scale)
    self.models = Models(model_list)
    self.x1, self.x2 = self.compute_x12()

  def next_scale(self):
    for i in xrange(self.step):
      scale = lambda x : self.x2 + float(x)*(self.x1-self.x2)/(self.step_range-1)
      model_list = self.build_model_list(scale)
      self.models = Models(model_list)
      self.x1, self.x2 = self.compute_x12()

  def run(self):
    self.first_scale()
    self.next_scale()

def truc(models):
  r = map(lambda x : (x.score,x.params["alpha"]),models.models)
  r.sort(key=lambda x : x[1])
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
    x1 = r[len(r)-2][1]
    x2 = r[len(r)-1][1]
  else:
    x1 = r[i_min-1][1]
    x2 = r[i_min+1][1]
    
  print x1
  print x2
  return x1,x2



