#!/usr/bin/env python

from settings               import *
from CustomModels           import *
from Globals                import *
from ModelExemple           import *



class Test:
  def __init__(self):
    alpha = 0.1
    model = OnlineLinearLearning
    self.model = smallModel(model)({"alpha" : alpha}) 

    self.expected    = 0.479075086746664269288231707832892425358295440673828125

  def run(self):
    self.model.train()
    self.model.validate()
    found = self.model.getValidationLogLoss()
    self.model.dump_score()

    if found == self.expected:
      print("We're Good !")
    else:
      print("Houston, we got a problem. Found : %.70f, Expected : %s" % (found, self.expected))
      print(found-self.expected)

  def run_multi(self):
    from Models import Models 
    model_list = []
    for i in range(num_cores):
      model_list.append(smallModel(pow(10,-1)))

    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()

    print("Test ended")

