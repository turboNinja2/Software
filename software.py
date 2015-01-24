from Models import *
from ModelExemple import *

class SoftwareTM():


  INIT_SCALE = {
      str(type(0.1)) : lambda x : pow(10,-x),
      str(type(1))   : lambda x : x,
  }

  SCALE = {
    str(type(0.1)) : lambda x1, x2, step_range : (lambda x : x2 + float(x+1)*(x1-x2)/(step_range+1)),
    str(type(1))   : lambda x1, x2, step_range : lambda x : x
  }
  
  def __init__(self,inf_bounds=None,sup_bounds=None,step=1,step_range=1,model=None,test=True,max_iterations=None):
    self.step       = step
    self.step_range = step_range
    self.inf_bounds = inf_bounds
    self.sup_bounds = sup_bounds
    self.max_iterations = max_iterations
    custom_kwargs = {}
    if max_iterations is not None:
      custom_kwargs = {"max_iterations":max_iterations}
    if test:
      self.model = smallModel(model,custom_kwargs)
    else:
      self.model = mediumModel(model,custom_kwargs)
    self.params = self.model.PARAMS_KEYS
    self.models = []
    self.result = []
    self.algo   = foo

  def compute_x12(self):
    self.result.extend(self.models.train_validated_dump_and_clear())
    self.inf_bounds, self.sup_bounds = self.algo(self.result)

  def build_model_list(self, scale_functions):
    model_list = []
    # Defining local vars
    param_number = len(self.params.keys())
    params_dict = {}
    step = self.step_range
    # Updating params dict
    for i in xrange(pow(step,param_number)):
      j = 0
      for param in self.params.keys():
        params_dict[param] = scale_functions[param](int(i/pow(step,j)) % step)
        j += 1
      model_list.append((self.model,{"params":dict(params_dict)}))

    return model_list

  def _build_init_scale(self):
    result = {}
    for param in self.params:
      result[param] = self.INIT_SCALE[str(self.params[param])]
    return result

  def _build_next_scale(self):
    result = {}
    for param in self.params:
      result[param] = self.SCALE[str(self.params[param])](self.inf_bounds[param],self.sup_bounds[param],self.step_range)
    return result


  def first_scale(self):
    init_scale = self._build_init_scale()
    model_list = self.build_model_list(init_scale)
    self.models = Models(model_list,True)
    self.compute_x12()


  def next_scale(self):
    for i in xrange(self.step):
      scale = self._build_next_scale()
      model_list = self.build_model_list(scale)
      del self.models
      self.models = Models(model_list,True)
      self.compute_x12()
      print "step %s done" % (i,)
      print self.result

  def run(self):
    if self.inf_bounds is None:
      self.first_scale()
    self.next_scale()
    print find_best_param(self.result)

def dichotomie(r):
  r.sort(key=lambda x : x[0])
  print(r)
  r_min = 10
  i_min = -1
  for i,value in enumerate(r):
    if value[0] < r_min:
      r_min = value[0]
      i_min = i
  if i_min == 0:
    x1 = r[0][0]
    x2 = r[1][0]
  elif i_min == len(r)-1:
    x1 = r[len(r)-2][0]
    x2 = r[len(r)-1][0]
  else:
    x1 = r[i_min-1][0]
    x2 = r[i_min+1][0]
    
  print(x1)
  print(x2)
  return x1,x2

def foo(list_score):
  #Finding the minimum
  #Fonctionne uniquement pour les fonctions convexes suivant toutes les variables
  inf_bounds = {}
  sup_bounds = {}
  best_score, best_params = find_best_param(list_score)


  for param_name in list_score[0][0].keys():
    x1, x2 = find_bounds(list_score, param_name,best_params[param_name])
    inf_bounds[param_name] = x1
    sup_bounds[param_name] = x2

  return inf_bounds, sup_bounds

def find_best_param(list_score):
  r_min = 100
  best_params = None
  for params, score in list_score:
    if score < r_min:
      r_min       = score
      best_params = params
  return r_min, best_params

 


def find_bounds(list_score,param_name,best_param):
  inf_bound = best_param
  sup_bound = best_param
  for param, score in list_score:
    v = param[param_name]
    if best_param == inf_bound and v < best_param:
      inf_bound = v
    if v < best_param and v > inf_bound:
      inf_bound = v
    if best_param == sup_bound and v > best_param:
      sup_bound = v
    if v > best_param and v < sup_bound:
      sup_bound = v
  return inf_bound, sup_bound
  
