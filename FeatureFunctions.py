from datetime import *

def cutHourAndDay(feat) :
  year  = 2000 + int(feat[0:2])
  month = int(feat[2:4])
  day   = int(feat[4:6])
  hour  = int(feat[6:8])
  dt    = datetime(year = year, month = month, day = day, hour = hour)
  dayOfWeek = dt.weekday()
  return [str(dayOfWeek), str(hour)]
