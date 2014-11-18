#!/usr/bin/env python

from test import Test


MULTI = False


t = Test()
t.run()
if MULTI:
  t.run_multi()

