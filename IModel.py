from math           import *
from DataOperations import *
from datetime       import datetime
from tools.misc     import *

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
    trainPath = self.trainPath
    to_dump = model_desc + ", Parser: %s, score: %s, logLoss: %s, trainingPath: %s\n" % (self.parser_mode,validationLogLoss,trainingLogLoss, trainPath )
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
    return self
 
  def validate(self):
    path = self.validationPath
    self.run_data(path,False)
    return self

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
    print("iterations : %s " % tt)

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
    submissionName = dummyString + '_Submission.csv' 
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


