from Globals import *

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, traindata=False):
  d = DataParser(path)
  d.run()
  """
    for t, line in enumerate(open(path)):
        if t == 0:
            x = [0] * 27
            continue
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif traindata and m == 1:
                y = float(feat)
            else:
                x[m] = abs(hash(str(m) + '_' + feat)) % D

        yield (ID, x, y) if traindata else (ID, x)
  """


class DataParser:
 
  ######################################################################
  ## PARSING METHOD

  def classic(self, line, x):
    for m, feat in enumerate(line.rstrip().split(',')):
      if m == 0:
        ID = int(feat)
      elif self.traindata and m == 1:
        y = float(feat)
      else:
        x[m] = abs(hash(str(m) + '_' + feat)) % D
    return (ID, x, y)

  def cross_prod(self, line, x):
    i = 0 
    for m1, feat1 in enumerate(line, rstrip().split(',')):
      if m1 == 0:
        ID1 = int(feat1)
      elif self.traindata and m1 == 1:
        y = float(feat1)
      else:
        for m2, feat2 in enumerate(line, rstrop().split(',')):
          if m2 != 0 and not (self.trandata and m2 == 1):
            x[i] = abs(hash(str(m1) + '_' + feat1 + '_' + str(m2) + '_' + feat2)) %D
            print len(line)
            i += 1
    return (ID, x, y)
            
        

 
  PARSING_METHODS = {
    "classic"     : classic,
    "cross_prod"  : cross_prod,
  }

  PARSING_LENGHT = {
    "classic"     : 27,
    "cross_prod"  : 27*27, 
  }
  ######################################################################
  ## CORE FUNCTIONS
  def __init__(self, path, traindata=False, mode = "classic"):
    self.path = path
    self.parsing_method = self.PARSING_METHODS[mode]
    self.parsing_lenght = self.PARSING_LENGHT[mode]
    self.traindata = traindata


  def run(self):
    for t, line in enumerate(open(self.path)):
      if t == 0:
        x = [0] *self.parsing_lenght
        continue
      ID, x, y = self.parsing_method(self, line, x)
    yield (ID, x, y) if traindata else (ID, x)

         
   





# The files contains 47 686 525 lines
def countLines(path):
    nbLines =0
    for t, line in enumerate(open(path)):
        nbLines +=1
    return(nbLines)

def createValidationSet(inputPath,filename):
    inputFile = inputPath + filename
    with open(inputPath + 'train_set.csv', 'w') as outfileTrain:
        with open(inputPath + 'validation_set.csv', 'w') as outfileValidation:
            for t, line in enumerate(open(inputFile)):
                if t == 0:
                    header = line
                    outfileTrain.write(header)
                    outfileValidation.write(header)
                    continue
                if t < 40000000:
                    outfileTrain.write(line)
                else:
                    outfileValidation.write(line)
