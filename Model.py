from math           import *
from DataOperations import *
from datetime       import datetime
from tools.misc     import shrink, copysign,  logloss

class Model:

  ####################################################################################
  ## INIT FUNCTIONS
  def __init__(self, params, wInit, trainPath=None,validationPath=None, refreshLine = None,
               nbZeroesParser = 2):
    self.params           = params
    self.w                = wInit
    self.nbIterations     = 0
    self.nbIterationsValidation     = 0
    self.loss             = 0
    self.validation_loss  = 0
    self.name             = "Unamed"
    self.trainPath        = trainPath
    self.validationPath   = validationPath
    self.refreshLine      = refreshLine
    self.nbZeroes         = nbZeroesParser
    self.score            = None
    for key in params.keys():
      setattr(self, key, params[key])

  def predict(self, x):
    wTx,_ = self.innerProduct(x)
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid
  
  def innerValidation(self,x,y):        
    p = self.predict(x)
    self.validation_loss += logloss(p,y)
    self.nbIterationsValidation += 1

  ####################################################################################
  ## GETTING FUCNTIONS

  def getLogLoss(self):
    if self.loss == 0:
      return 0
    return self.loss * 1. / self.nbIterations

  def getValidationLogLoss(self):
    if self.validation_loss == 0:
      return 0
    return self.validation_loss * 1. / self.nbIterationsValidation

  def get_score(self):
    if self.score is None:
      raise Exception("Not computed yet")
    return self.score
 
  def dumping_string(self):
    model_desc = str(self)
    score = self.get_score()
    to_dump = model_desc + ", score : %s\n" % (score,)
    return to_dump


  ####################################################################################
  ## RUNNING FUNCTIONS


  def train(self):
    path = self.trainPath
    self.run_data(path,update=True)
 
  def validate(self):
    path = self.validationPath
    self.score = self.run_data(path,False)
    return self.score

  def run_data(self, path,update=False):
    tt = 1
    data = DataParser(path)
    validation_loss = 0
    for ID, x, y in data.run():
      if update:
        self.update(x, y)
      else:
        self.innerValidation(x,y)
      self.refreshed(tt,update)
      tt += 1
    return self.validation_loss * 1. / tt

  def refreshed(self, tt, update):
    if tt % self.refreshLine == 0:
      print('Model desc:' + str(self))
      if update:
        print('%s\tencountered: %d\t training loss: %f' % (datetime.now(), tt, self.getLogLoss()))
      else:
        print('%s\tencountered: %d\t validation loss: %f' % (datetime.now(), tt, self.getValidationLogLoss()))
  ####################################################################################
  ## PRINTING FUNCTIONS

  def __str__(self):
   return "%s, params : %s" % (self.name, str(self.params))

  def dump_score(self,output_path):
    dumping_string = self.dumping_string()
    try:
      f = open(output_path,'a')
      f.write(dumping_string)
      f.close()
    except:
      f = open(str(self),'a')
      f.write(dumping_string)
      f.close()
    
  ####################################################################################
  ## CORE FUNCTIONS

  def innerProduct(self,x):
    wTx = 0.
    n = 0
    for i in x:  # do wTx
      wTx += self.w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
      n+=1
    return wTx,n

  def predict(self, x):
    wTx,_ = self.innerProduct(x)
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid

  def update(self,x,y):
    self.pretrement(x)
    p = self.predict(x)
    self.nbIterations += 1
    self.loss += logloss(p,y)
    self.loop(p,x,y)

  def pretrement(self,x):
    pass   

  def loop(self,p,y,x):
    raise Exception("Undefined method loop")

######################################################################################
## CUSTOM MODELS


class OnlineLinearLearning(Model):
  def __init__(self,params,wInit,**kwargs):
    Model.__init__(self,params, wInit,**kwargs)
    self.n = [0] * len(wInit)
    self.name = "Online method"

  def loop(self,p,x,y):
    for i in x:
      self.n[i] += abs(p - y)
      self.w[i] -= (p - y) * 1. * self.alpha / sqrt(self.n[i])

class ZALMS(Model):
  def __init__(self,params,wInit,**kwargs):
    Model.__init__(self,params, wInit,**kwargs)
    self.name = "ZALMS"

  def loop(self,p,x,y):
    for i in x:
      self.w[i] -= self.delta * ((p - y) * 1. + self.rho * copysign(self.w[i],1))

class OLBI(Model):
  def __init__(self,params,wInit,**kwargs):
    Model.__init__(self,params, wInit,**kwargs)
    self.name = "OLBI"

  def loop(self,p,x,y):
    for i in x:
      self.w[i] -= self.delta * (p - y) * 1. 
      self.w[i] = shrink(self.w[i], self.gamma) 

class PA(Model):
  def __init__(self,params,wInit):
    Model.__init__(self,params, wInit)
    self.name = "PA"

  def update(self, x, y):
    yBis = 2 * y - 1
    wTx,n = self.innerProduct(x)
    sufferLoss = max(0,1 - yBis * wTx)
    self.nbIterations += 1
    p = copysign(1,wTx)
    self.loss += logloss((p + 1.) / 2., y)  # for progressive validation
    tau = sufferLoss / n
    for i in x:
      self.w[i] += tau * yBis * 1.  

  def predict(self,x):
    wTx,_ = self.innerProduct(x)
    p = (1 + copysign(1,wTx)) / 2.
    return p

class PAI(Model):
  def __init__(self,params,wInit):
    Model.__init__(self,params, wInit)
    self.name = "PA-I"

  def update(self, x, y):
    yBis = 2 * y - 1
    wTx,n = self.innerProduct(x)
    sufferLoss = max(0,1 - yBis * wTx)
    self.nbIterations += 1
    p = copysign(1,wTx)
    self.loss += logloss((p + 1.) / 2., y)  # for progressive validation
    tau = min(self.C,sufferLoss / n)
    for i in x:
      self.w[i] -= tau * yBis * 1. 

  def predict(self,x):
    wTx,_ = self.innerProduct(x)
    p = (1 + copysign(1,wTx)) / 2.
    return p
