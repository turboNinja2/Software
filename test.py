#!/usr/bin/env python

from settings               import *
from Model                  import *
from Globals                import *
from ModelExemple           import smallModel

class Test:
  def __init__(self):
    self.model = smallModel() 

    self.expected    = 0.167259060835432709423848

  def run(self):
    self.model.train()
    found = self.model.validate()
    self.model.dump_score()

    if found == self.expected:
      print("We're Good !")
    else:
      print("Houston, we got a problem. Found : %s, Expected : %s" % (found, self.expected))

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

