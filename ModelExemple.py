from Model    import *
from settings import * 


def smallModel(alpha=0.1):
  Learning    = OnlineLinearLearning

  train       = dataPath + "small_train_set.csv"
  validation  = dataPath + "small_validation_set.csv"
  dump        = dataPath + "results/test_results.csv"
  json_dump   = dataPath + "results/test_json_results.csv" 

  kwargs = {
    "trainPath"       : train,
    "validationPath"  : validation,
    "dumpingPath"     : dump,
    "jsonDumpingPath" : json_dump,
    "refreshLine"     : 75,
  }

  params = {"alpha":alpha}
  w = [0.] * D

  model = Learning(params, w, **kwargs)
  return model

def mediumModel(alpha=0.1):
  Learning    = OnlineLinearLearning

  train       = dataPath + 'medium_train_set.csv'
  validation  = dataPath + 'medium_validation_set.csv'
  dump        = dataPath + "results/medium_results.csv"
  json_dump   = dataPath + "results/medium_json_results.csv"

  kwargs = {
    "trainPath"       : train,
    "validationPath"  : validation,
    "dumpingPath"     : dump,
    "jsonDumpingPath" : json_dump,
    "refreshLine"     : 1000000,
    "parser_mode"     : "classic2",
  }

  params = {"alpha" : alpha}
  w = [0.] * D

  model = Learning(params, w, **kwargs)

  return model
