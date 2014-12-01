from Models import *

class SoftwareTM():
  
  def __init__(self,xmin=None,xmax=None,model=None,step=1,step_range=1):
    self.step       = step
    self.step_range = step_range
    self.x1 = xmin
    self.x2 = xmax
    if model is None:
      self.model = smallModel
    else:
      self.model = model
    self.models = []
    self.result = []
    self.algo   = dichotomie

  def compute_x12(self):
    self.models.train()
    self.models.validation()
    self.result.extend(map(lambda x : (x.alpha, x.getValidationLogLoss()), self.models.models))
    x1,x2 = self.algo(self.result)
    return x1, x2

  def build_model_list(self, scale_function):
    model_list = []
    for i in xrange(self.step_range):
      model_list.append(self.model(scale_function(i)))
    return model_list

  def first_scale(self):
    init_scale = lambda x : pow(10,-x)
    model_list = self.build_model_list(init_scale)
    self.models = Models(model_list)
    self.x1, self.x2 = self.compute_x12()
    self.models.dump()


  def next_scale(self):
    for i in xrange(self.step):
      scale = lambda x : self.x2 + float(x+1)*(self.x1-self.x2)/(self.step_range+1)
      model_list = self.build_model_list(scale)
      del self.models
      self.models = Models(model_list)
      self.x1, self.x2 = self.compute_x12()
      self.models.dump()
      print "step %s done" % (i,)
      print self.result

  def run(self):
    if self.x1 is None:
      self.first_scale()
    self.next_scale()

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



