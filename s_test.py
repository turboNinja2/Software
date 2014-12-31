#!/usr/bin/env python


from software import *
from algo_settings import software_kwargs as kwargs

if __name__ == "__main__" :
  s = SoftwareTM(**kwargs)
  s.run()
