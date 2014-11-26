from math           import *
from DataOperations import *
from datetime       import datetime
from tools.misc     import shrink, copysign,  logloss

import csv
import json

class Model:

  ####################################################################################
  ## INIT FUNCTIONS

  def __init__(self, params, wInit, 
                parser_mode="classic",
                **kwargs):
    self.params = params
    self.w = wInit

    self.nbIterationsTraining = 0 
    self.nbIterationsValidation = 0

    self.loss = 0
    self.validation_loss = 0

    self.name = "Unamed"

    self.parser_mode = parser_mode

    for key in params.keys():
      setattr(self, key, params[key])

    for key in kwargs.keys():
      setattr(self, key, kwargs[key])

    try:
      kwargs["max_iterations"]
    except:
      self.max_iterations = None

  ####################################################################################
  ## GETTING FUCNTIONS

  def getTrainingLogLoss(self):
    if self.loss == 0:
      return 0
    return self.loss * 1. / self.nbIterationsTraining

  def getValidationLogLoss(self):
    if self.validation_loss == 0:
      return 0
    return self.validation_loss * 1. / self.nbIterationsValidation

  ####################################################################################
  ## PRINTING FUNCTIONS

  def dumping_string(self):
    model_desc = str(self)
    validationLogLoss = self.getValidationLogLoss()
    trainingLogLoss = self.getTrainingLogLoss()
    to_dump = model_desc + ", Parser Mode : %s, score : %s, logLoss : %s\n" % (self.parser_mode,validationLogLoss,trainingLogLoss)
    return to_dump

  def dumping_dict(self):
    name = self.name
    params = self.params
    validationLogLoss = self.getValidationLogLoss()
    trainingLogLoss = self.getTrainingLogLoss()
    parser_mode = self.parser_mode
    dict_to_dump = {
      "name"        : name,
      "params"      : params,
      "validationLogLoss"       : validationLogLoss,
      "trainingLogLoss"     : trainingLogLoss,
      "parser_mode" : parser_mode,
    }
    return dict_to_dump

  def dumping_list(self):
    name = self.name
    params = self.params
    validationLogLoss = self.getValidationLogLoss()
    trainingLogLoss = self.getTrainingLogLoss()
    parser_mode = self.parser_mode
    list_to_dump = [name,params,validationLogLoss,trainingLogLoss,parser_mode]
    return list_to_dump

  def __str__(self):
   return "%s, params : %s" % (self.name, str(self.params))

  def dump_score(self):
    dumping_string = self.dumping_string()
    dumping_dict = self.dumping_dict()
    dumping_list = self.dumping_list()
    json_dict = json.dumps(dumping_dict)
    #try:
    f = open(self.dumpingPath,'a')
    json_f = open(self.jsonDumpingPath, 'a')
    a = csv.writer(f)
    a.writerow(dumping_list)
    json_f.write(json_dict)
    f.close()
    """
    except:
      f = open(str(self),'a')
      f.write(dumping_string)
      f.close()
    """
 
  ####################################################################################
  ## RUNNING FUNCTIONS

  def train(self):
    path = self.trainPath
    self.run_data(path,update=True)
 
  def validate(self):
    path = self.validationPath
    self.run_data(path,False)

  def run_data(self, path,update=False):
    tt = 1
    data = DataParser(path,mode = self.parser_mode)
    validation_loss = 0
    for ID, x, y in data.run():
      if update:
        self.update(x, y)
      else:
        self.innerValidation(x,y)
      self.refreshed(tt,update)
      tt += 1
      if self.max_iterations is not None and self.max_iterations == tt:
        break
    print "iterations : %s " % tt

  def refreshed(self, tt, update):
    if tt % self.refreshLine == 0:
      print('Model desc:' + str(self))
      if update:
        print('%s\tencountered: %d\t training loss: %f' % (datetime.now(), tt, self.getTrainingLogLoss()))
      else:
        print('%s\tencountered: %d\t validation loss: %f' % (datetime.now(), tt, self.getValidationLogLoss()))

  def writeSubmission(self):
    dt = datetime.now().__str__()
    dummyString = ''.join(e for e in dt if e.isalnum())
    submissionName = dummyString +'_Submission.csv' 
    descriptionName = dummyString + '_Description.txt'

    with open(self.submissionPath + submissionName, 'w') as outfile:
      data = DataParser(self.testPath,mode = self.parser_mode)
      outfile.write('id,click\n')
      for ID, x,y  in data.run():
        p = self.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))
  
    with open(self.submissionPath + descriptionName, 'w') as outfile:
      outfile.write(self.dumping_string())

   
  ####################################################################################
  ## CORE FUNCTIONS
 
  def innerValidation(self,x,y):        
    self.pretreatment(x)
    p = self.predict(x)
    self.validation_loss += logloss(p,y)
    self.nbIterationsValidation += 1

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
    self.pretreatment(x)
    p = self.predict(x)
    self.nbIterationsTraining += 1
    self.loss += logloss(p,y)
    self.loop(p,x,y)

  def pretreatment(self,x):
    pass   

  def loop(self,p,y,x):
    raise Exception("Undefined method loop")

######################################################################################
## CUSTOM MODELS
class OnlineLinearLearning(Model):
  def __init__(self,params,wInit,**kwargs):
    Model.__init__(self,params, wInit, **kwargs)
    self.n = [0] * len(wInit)
    self.name = "Online method"

  def loop(self,p,x,y):
    for i in x:
      self.n[i] += abs(p - y)
      self.w[i] -= (p - y) * 1. * self.alpha / sqrt(self.n[i])

class LogOnlineLinearLearning(Model):
  def __init__(self,params,wInit,**kwargs):
    Model.__init__(self,params, wInit, **kwargs)
    self.n = [0] * len(wInit)
    self.name = "Log Online method"

  def loop(self,p,x,y):
    for i in x:
      self.n[i] += abs(p - y)
      self.w[i] -= ((1 - y) / (1 - p) - y / p) * 1. * self.alpha / sqrt(self.n[i])

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

class Perceptron(Model):
  def __init__(self,params,wInit,**kwargs):
    Model.__init__(self,params, wInit,**kwargs)
    self.name = "Perceptron"

  def predict(self,x):
    wTx,_ = self.innerProduct(x)
    p = sigmoid(self.approx * wTx)
    return p

  def loop(self,p,x,y):
    if (y - 0.5) * (p - 0.5) <= 0: # if the predictions disagree
      for i in x: # contribution of each feature is corrected
        self.w[i] += (y - 0.5) * 2.

class Amazing(Model):
  def __init__(self,params, wInit,**kwargs):
    Model.__init__(self,params, wInit,**kwargs)
    self.name = "Amazing Algorithm"

  def predict(self, x):
    return 0.17
  
  def loop(self,p,x,y):
    pass
