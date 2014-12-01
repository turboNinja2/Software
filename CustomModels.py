from math           import *
from DataOperations import *
from datetime       import datetime
from tools.misc     import *
from IModel import *
from random import random

######################################################################################
## CUSTOM MODELS
class OnlineLinearLearning(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.n = [0] * D
    self.name = "Online method"

  def loop(self,p,x,y):
    for i in x:
      self.n[i] += abs(p - y)
      self.w[i] -= (p - y) * 1. * self.alpha / sqrt(self.n[i])

class LogOnlineLinearLearning(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.n = [0] * D
    self.name = "Log Online method"

  def loop(self,p,x,y):
    for i in x:
      self.n[i] += abs(p - y)
      self.w[i] -= max(min((1 - y) / (1 - p) - y / p,10 ** 8),-10 ** 8) * 1. * self.alpha / sqrt(self.n[i])

class ZALMS(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.name = "ZALMS"

  def loop(self,p,x,y):
    for i in x:
      self.w[i] -= self.delta * ((p - y) * 1. + self.rho * copysign(self.w[i],1))

class OLBI(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.name = "OLBI"

  def loop(self,p,x,y):
    for i in x:
      self.w[i] -= self.delta * (p - y) * 1. 
      self.w[i] = shrink(self.w[i], self.gamma) 

class Perceptron(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.name = "Perceptron"

  def predict(self,x):
    wTx,_ = self.innerProduct(x)
    p = sigmoid(self.approx * wTx)
    return p

  def loop(self,p,x,y):
    if (y - 0.5) * (p - 0.5) <= 0: # if the predictions disagree
      for i in x: # contribution of each feature is corrected
        self.w[i] += (y - 0.5) * 2.

class Perceptron2(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.name = "Perceptron2"

  def predict(self,x):
    wTx,_ = self.innerProduct(x)
    p = sigmoid(self.approx * wTx)
    return p

  def loop(self,p,x,y):
    if (y - 0.5) * (p - 0.5) <= 0: # if the predictions disagree
      for i in x: # contribution of each feature is corrected
        self.w[i] += min(max(-1, (y - 0.5) * 2.),1)

class RandomNeuralNetwork(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.name = "RandomNeuralNetwork"
    self.neurons = []
    self.signals = []
    self.n = []

  def neuronActivate(self,neuron,x):
      res = 0
      for i in x :
          self.w[i] = 1
      for n in neuron:
        res = res + self.w[n]
      for i in x :
          self.w[i] = 0
      return res > self.threshold

  def predict(self, x) :
      sumSignals = 0
      nbNeurons = len(self.neurons)
      if(nbNeurons > 0):
          for i in range(nbNeurons) :
              isActive = self.neuronActivate(self.neurons[i],x)
              if isActive :
                  sumSignals += self.signals[i]
          return (sumSignals * 1. / (2 * nbNeurons) + 0.5)
      else :
        return 0.5

  def loop(self,p,x,y):
      if (random() < 0.0001) and (len(self.neurons) < self.maxNeurons):
          self.neurons.append(x)
          if len(self.neurons) == self.maxNeurons :
             print("Maximum number of neurons reached")
          self.signals.append((y - 0.5) * 2)

class FTRLProximal(Model):
  def __init__(self,params,**kwargs):
    Model.__init__(self,params,**kwargs)
    self.name = "FTRLProximal"
    self.z = [0] * D
    self.sigma = [0] * D
    self.g = [0] * D
    self.n = [0] * D

  def predict(self,x):
    wTx,_ = self.innerProduct(x)
    p = sigmoid(self.approx * wTx)
    return p

  def loop(self,p,x,y):
    error = (p - y)
    for i in x: # contribution of each feature is corrected
      if copysign(self.z[i],1) <= self.lambda1:
        self.w[i] = 0
      else:
        self.w[i] = (self.lambda1 * copysign(1,self.z[i]) - self.z[i]) / (((self.beta + sqrt(self.n[i])) / self.alpha) + self.lambda2)

      for i in x :
        self.g[i] = error * 1.
        self.sigma[i] = 1 / self.alpha * (sqrt(self.n[i] + self.g[i] ** 2) - sqrt(self.n[i])) 
        self.z[i] += self.g[i] - self.sigma[i] * self.w[i]
        self.n[i] += self.g[i] ** 2
