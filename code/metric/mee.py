import numpy as np

def MEE(Y,d):
  if len(Y.shape)==1 or len(d.shape)==1:
    return np.mean(np.abs(Y-d))
  square_distance = np.sum(np.square(Y-d),axis=1)
  return np.mean(np.sqrt(square_distance))