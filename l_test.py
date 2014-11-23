#!/usr/bin/env python

from test import Test

if __name__ == "__main__":
  MULTI = False
  t     = Test()
  t.run()
  if MULTI:
    t.run_multi()