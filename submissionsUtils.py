import pandas as pd

def average(path,files):
  nbFiles = len(files)
  pdFiles = [] 
  for i in range(nbFiles):
    pdFiles.append(pd.read_csv(path + files[i]))
  sum = pdFiles[0]["click"]
  resName = files[0]
  for i in range(1,nbFiles):
    sum = sum + pdFiles[i]["click"]
    resName = resName + files[i]

  sum = sum / nbFiles
  result = {"id":  pdFiles[0]["id"], "click": sum}
  c = ["id", "click"]
  df = pd.DataFrame(data=result,columns  = c)
  df.to_csv(path + resName, header = True, index = False)


dataPath = 'C:/Users/JUJulien/Desktop/KAGGLE/Competitions/Avazu/Data/'
submissionPath = dataPath + 'submissions/'

average(submissionPath,['20141126094442231000_Submission.csv',
                        '20141123193248753000_Submission.csv',
                        '20141125202949615000_Submission.csv'])