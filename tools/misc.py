from math import  copysign, log, exp

def shrink(z,gamma):
  absz = copysign(z,1)
  return copysign(max(0, absz - gamma),z)

def logloss(p, y):
  p = max(min(p, 1. - 10e-15), 10e-15)
  return -log(p) if y == 1. else -log(1. - p)

def sigmoid(a):
  return 1. / (1. + exp(-max(min(a, 20.), -20.)))  # bounded sigmoid
