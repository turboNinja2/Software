#!/usr/bin/env python

from test import Test

MULTI = True
t = Test()
t.run()
if MULTI:
  t.run_multi()

