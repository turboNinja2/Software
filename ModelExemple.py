from Model    import *
from settings import dataPath 
from datetime import datetime 
 
dt = datetime.now().__str__()
dummyString = ''.join(e for e in dt if e.isalnum())


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
  Learning    = LogOnlineLinearLearning

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

def realModel(alpha=0.01):
  Learning = LogOnlineLinearLearning

  train       = dataPath + 'train_set.csv'  # path to training file
  validation  = dataPath + 'validation_set.csv'  # path to testing file
  dump        = dataPath + "results/results" + dummyString + ".csv"
  json_dump   = dataPath + "results/json_results" + dummyString + ".csv"

  kwargs = {
    "trainPath"       : train,
    "validationPath"  : validation,
    "dumpingPath"     : dump,
    "jsonDumpingPath" : json_dump,
    "refreshLine"     : 2.5*pow(10,6),
    "parser_mode"     : "classic2",
  }

  params = {"alpha" : alpha}
  w = [0.] * D

  model = Learning(params, w, **kwargs)
   
  return model
