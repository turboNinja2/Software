#!/usr/bin/env python

from test import Test
from datetime import datetime
from timeit import timeit
from time import sleep
from time import time

if __name__ == "__main__":
  MULTI = False
  t     = Test()
  t.run()
  if MULTI:
    t.run_multi()
