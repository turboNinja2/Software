from math import *



class Model:

  def __init__(self, params, wInit):
    self.params = params
    self.w = wInit


class OnlineLinearLearning(Model):
  def __init__(self,params,wInit):
    self.alpha = params[0]
    self.w = wInit

  def description():
    return 'On line method, alpha =' + self.alpha 

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
  def update(self, n, x, y):
    p = self.predict(x)
    for i in x:
      # alpha / sqrt(n) is the adaptive learning rate
      # (p - y) * x[i] is the current gradient
      # note that in our case, if i in x then x[i] = 1.
      n[i] += abs(p - y)
      self.w[i] -= (p - y) * 1. * self.alpha / sqrt(n[i])

class ZALMS(Model):
  def __init__(self,params,wInit):
    self.delta = params[0]
    self.rho = params[1]
    self.w = wInit

  def description():
    return 'ZALMS, delta =' + self.delta + 'rho = ' + self.rho

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
  
  def update(self, n, x, y):
    p = self.predict(x)
    for i in x:
      n[i] += abs(p - y)
      self.w[i] -= self.delta * ((p - y) * 1. + self.rho * copysign(self.w[i],1))

