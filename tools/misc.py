from math import  copysign

def shrink(z,gamma):
  absz = copysign(z,1)
  return copysign(max(0, absz - gamma),z)