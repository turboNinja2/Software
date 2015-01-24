from CustomModels    import *
from IModel          import Model
from settings import dataPath 
from datetime import datetime 
 
dt = datetime.now().__str__()
dummyString = ''.join(e for e in dt if e.isalnum())


DEFAULT_MODEL = OnlineLinearLearning

TRAIN       = dataPath + "train_set.csv"
VALIDATION  = dataPath + "validation_set.csv"
RESULT_PATH = dataPath + "results/"

small_kwargs = {
      "trainPath"       : TRAIN,
      "validationPath"  : VALIDATION,
      "dumpingPath"     : RESULT_PATH + "test_results.csv",
      "jsonDumpingPath" : RESULT_PATH + "test_json_results.csv",
      "refreshLine"     : 75,
      "max_iterations"  : 100,
    }

medium_kwargs = {
      "trainPath"       : TRAIN,
      "validationPath"  : VALIDATION,
      "dumpingPath"     : RESULT_PATH + "medium_result.csv",
      "jsonDumpingPath" : RESULT_PATH + "medium_json_result.csv",
      "refreshLine"     : pow(10,6),
      "parser_mode"     : "classic2",
      "max_iterations"  : 2*pow(10,6)
    }

real_kwargs = {
      "trainPath"       : TRAIN,
      "validationPath"  : VALIDATION,
      "dumpingPath"     : RESULT_PATH + "results" + dummyString + ".csv",
      "jsonDumpingPath" : RESULT_PATH + "results" + dummyString + ".csv",
      "refreshLine"     : 2.5*pow(10,6),
      "parser_mode"     : "classic2",
    }


 

def model_builder(model,model_kwargs):
  mo = model

  mo.custom_init = model.__init__
  def new_init(self,params,**kwargs):
    #kwargs.update(model_kwargs)
    for key in model_kwargs.keys():
      if key not in kwargs:
        kwargs[key] = model_kwargs[key]
    self.custom_init(params,**kwargs)
  mo.__init__ = new_init
  return mo

def smallModel(model=DEFAULT_MODEL,custom_kwargs={}):
  kwargs = small_kwargs
  kwargs.update(custom_kwargs)
  return model_builder(model,kwargs)

def mediumModel(model=DEFAULT_MODEL,custom_kwargs={}):
  kwargs = medium_kwargs
  kwargs.update(custom_kwargs)
  return model_builder(model,kwargs)

def realModel(model=DEFAULT_MODEL,custom_kwargs={}):
  kwargs = real_kwargs
  kwargs.update(custom_kwargs)
  return model_builder(model,kwargs)

###################################################################
## HOW TO
"""
How to use the Model Exemples.

small models are use for tests, medium for software and real model to the real algorithm.

you can call small model with the model you want, and some kwargs you want.
This will create a Model genetor.
Use the model generator as a custom class to call models. Some paths, and several other params as been already computed, so you don't have to deal with it.

here's an exemple : 
my_cute_model_generator = smallModel(model=OnlineLinearLearning,{max_iterations:200})

know, if you got some kwargs, you can create a small_model like this :
my_model = my_cute_model_generator({"params":{"alpha":0.1}})
Amazing, isn't it ?

questions ? ulysseklatzmann@gmail.com
"""

