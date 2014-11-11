from math import *
from ErrorEvaluation import logloss
from DataOperations import *
from datetime import datetime
from tools.misc import shrink, copysign

class Model:
  def __init__(self, params, wInit, nbZeroesParser = 2):
    self.params = params
    self.w = wInit
    self.nbIterations = 0
    self.loss = 0
    self.validation_loss = 0
    self.name = "Unamed"
    self.nbZeroes = nbZeroesParser
    for key in params.keys():
      setattr(self, key, params[key])

  def __str__(self):
   return "%s, params : %s" % (self.name, str(self.params))
 
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
  
  def getLogLoss(self):
    if self.loss == 0:
      return 0
    return self.loss * 1. / self.nbIterations

  def run(self, trainPath,update=True,customRefreshLine=None):
    global refreshLine
    if customRefreshLine is not None:
      refreshLine = customRefreshLine

    tt = 1
    data = DataParser(trainPath,nbZeroes = self.nbZeroes)
    self.validation_loss = 0
    for ID, x, y in data.run():
      if update: 
        self.update(x, y)
      else:
        p = self.predict(x)
        self.validation_loss += logloss(p,y)
      # print out progress, so that we know everything is working
      if tt % refreshLine == 0:
        print('Model desc:' + str(self))
        print('%s\tencountered: %d\t logloss: %f' % (datetime.now(), tt, self.getLogLoss()))
      tt += 1
    if not update:
      return self.validation_loss * 1. / tt

class OnlineLinearLearning(Model):
  def __init__(self,params,wInit):
    Model.__init__(self,params, wInit)
    self.n = [0] * len(wInit)
    self.name = "Online method"

  def update(self, x, y):
    p = self.predict(x)
    self.nbIterations += 1
    self.loss += logloss(p, y)  # for progressive validation
    for i in x:
      self.n[i] += abs(p - y)
      self.w[i] -= (p - y) * 1. * self.alpha / sqrt(self.n[i])

class ZALMS(Model):
  def __init__(self,params,wInit):
    Model.__init__(self,params, wInit)
    self.name = "ZALMS"

  def update(self, x, y):
    p = self.predict(x)
    self.nbIterations += 1
    self.loss += logloss(p, y)  # for progressive validation
    for i in x:
      self.w[i] -= self.delta * ((p - y) * 1. + self.rho * copysign(self.w[i],1))

class OLBI(Model):
  def __init__(self,params,wInit):
    Model.__init__(self,params, wInit)
    self.name = "OLBI"

  def update(self, x, y):
    p = self.predict(x)
    self.nbIterations += 1
    self.loss += logloss(p, y)  # for progressive validation
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
