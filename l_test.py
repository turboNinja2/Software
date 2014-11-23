#!/usr/bin/env python

from test import Test

if __name__ == "__main__":
  MULTI = True
  t     = Test()
  t.run()
  if MULTI:
    t.run_multi()