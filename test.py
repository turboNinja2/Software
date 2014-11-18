#!/usr/bin/env python

from settings               import *
from Model                  import *
from Globals                import *

class Test:


  def __init__(self):
    self.Learning    = OnlineLinearLearning

    train       = dataPath + "small_train_set.csv"
    validation  = dataPath + "small_validation_set.csv"
    dump        = dataPath + "results/test_results.csv"
    json_dump   = dataPath + "results/test_json_results.csv" 

    self.kwargs = {
      "trainPath"       : train,
      "validationPath"  : validation,
      "dumpingPath"     : dump,
      "jsonDumpingPath" : json_dump,
      "refreshLine"     : 150,
    }
    
    self.params = {"alpha":0.1}
    self.w = [0.] * D


    self.expected    = 0.167259060835432709423848

  def run(self):
    model = self.Learning(self.params, 
                     self.w, 
                     **self.kwargs
                     )
    model.train()
    found = model.validate()
    model.dump_score()

    if found == self.expected:
      print("We're Good !")
    else:
      print("Houston, we got a problem. Found : %s, Expected : %s" % (found, self.expected))

  def run_multi(self):
    from Models import Models 
    model_list = []
    for i in range(num_cores):
      model_list.append(self.Learning({"alpha" : 10 ** (-i)}, 
                    self.w,
                    **self.kwargs
                    ))

    models = Models(model_list)
    models.train()
    models.validation()
    models.dump()

    print("Test ended")
