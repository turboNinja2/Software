#!/usr/bin/env python


from software import *
from software_settings import kwargs

if __name__ == "__main__" :
  print kwargs
  s = SoftwareTM(**kwargs)
  s.run()
