import pandas as pd
from os import listdir

from datetime       import datetime

def average(path,files,descs):
  dt = datetime.now().__str__()
  nbFiles = len(files)
  pdFiles = [] 
  for i in range(nbFiles):
    pdFiles.append(pd.read_csv(path + files[i]))
    print(i)
  sum = pdFiles[0]["click"]
  resName = files[0]
  for i in range(1,nbFiles):
    sum = sum + pdFiles[i]["click"]
    resName = resName + files[i]

  sum = sum / nbFiles
  result = {"id":  pdFiles[0]["id"], "click": sum}
  c = ["id", "click"]
  df = pd.DataFrame(data=result,columns  = c)

  dummyString = ''.join(e for e in dt if e.isalnum())

  df.to_csv(path + dummyString + "_Avg.csv" , header = True, index = False)

  with open(path + dummyString + "_Desc.txt", 'w') as outfile:
    for i in range(nbFiles):
      for t, line in enumerate(open(path + descs[i])):
        outfile.write(line)


dataPath = 'C:/Users/JUJulien/Desktop/KAGGLE/Competitions/Avazu/Data/'
submissionPath = dataPath + 'submissions/calibration/'

submissions = []

filenames = listdir(submissionPath)
submissions = []
descriptions = []
for filename in filenames :
  if filename.endswith('.csv'):
    submissions.append(filename)
  elif filename.endswith('.txt'):
    descriptions.append(filename)

average(submissionPath,submissions,descriptions)