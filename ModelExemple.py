from CustomModels    import *
from settings import dataPath 
from datetime import datetime 
 
dt = datetime.now().__str__()
dummyString = ''.join(e for e in dt if e.isalnum())

model_exemple = OnlineLinearLearning
real_model = LogOnlineLinearLearning


class smallModel(model_exemple):


  def __init__(self, alpha=0.1,**kwargs):

    train       = dataPath + "train_set.csv"
    validation  = dataPath + "validation_set.csv"
    dump        = dataPath + "results/test_results.csv"
    json_dump   = dataPath + "results/test_json_results.csv" 
    submissionPath   = dataPath + "results/test_submission.csv" 

    kwargs.update({
      "trainPath"       : train,
      "submissionPath" : submissionPath,
      "testPath" : train,
      "validationPath"  : validation,
      "dumpingPath"     : dump,
      "jsonDumpingPath" : json_dump,
      "refreshLine"     : 75,
      "max_iterations"  : 100,
    })
    if "params" not in kwargs.keys():
      params = {"alpha":alpha}
      kwargs["params"] = params

    model_exemple.__init__(self,**kwargs)


class mediumModel(model_exemple):

  def __init__(self,alpha=0.1,**kwargs):

    train       = dataPath + 'test.csv'
    validation  = dataPath + 'test.csv'
    dump        = dataPath + "results/medium_results.csv"
    json_dump   = dataPath + "results/medium_json_results.csv"

    kwargs.update({
      "trainPath"       : train,
      "validationPath"  : validation,
      "dumpingPath"     : dump,
      "jsonDumpingPath" : json_dump,
      "refreshLine"     : 1000000,
      "parser_mode"     : "classic2",
      "max_iterations"  : 2*pow(10,6)
    })

    params = {"alpha" : alpha}

    model_exemple.__init__(self,params,**kwargs)


class realModel(real_model):


  def __init__(self,alpha=0.01,**kwargs):

    train       = dataPath + 'small_train_set.csv'  # path to training file
    validation  = dataPath + 'validation_set.csv'  # path to testing file
    dump        = dataPath + "results/results" + dummyString + ".csv"
    json_dump   = dataPath + "results/json_results" + dummyString + ".csv"

    kwargs.update({
      "trainPath"       : train,
      "validationPath"  : validation,
      "dumpingPath"     : dump,
      "jsonDumpingPath" : json_dump,
      "refreshLine"     : 2.5*pow(10,6),
      "parser_mode"     : "classic2",
    })

    params = {"alpha" : alpha}

    real_model.__init__(self,params,**kwargs)

