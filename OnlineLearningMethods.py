from math import *
from ErrorEvaluation import logloss

class Model:

  def __init__(self, params, wInit):
    self.params       = params
    self.w            = wInit
    self.nbIterations = 0
    self.loss         = 0
    self.name         = "Unamed"
    for key in params.keys():
      setattr(self, key, params[key])

  def __str__(self):
   return "%s, params : %s" % (self.name, str(params))
 
  # C.  Get probability estimation on x
  # INPUT:
  #   x: features
  #   w: weights
  # OUTPUT:
  #   probability of p(y = 1 | x; w)
  def predict(self, x):
    wTx = 0.
    for i in x:  # do wTx
      wTx += (self.w[i]) * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


  def getLogLoss(self):
    return self.loss * 1. /  self.nbIterations
    

class OnlineLinearLearning(Model):
  def __init__(self,params,wInit):
    Model.__init__(self,params, wInit)
    self.n      = [0] * len(wInit)
    self.name   = "Online method"


  ## D.  Update given model
  # INPUT:
  # alpha: learning rate
  #   w: weights
  #   n: sum of previous absolute gradients for a given feature
  #    this is used for adaptive learning rate
  #   x: feature, a list of indices
  #   p: prediction of our model
  #   y: answer
  # MODIFIES:
  #   w: weights
  #   n: sum of past absolute gradients
  def update(self, x, y):
    p = self.predict(x)
    self.nbIterations += 1
    self.loss += logloss(p, y)  # for progressive validation
    for i in x:
      # alpha / sqrt(n) is the adaptive learning rate
      # (p - y) * x[i] is the current gradient
      # note that in our case, if i in x then x[i] = 1.
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
