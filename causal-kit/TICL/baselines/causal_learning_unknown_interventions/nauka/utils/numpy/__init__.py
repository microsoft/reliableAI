# -*- coding: utf-8 -*-

from   .  import random



def blasTest(k=8192, dtype="float32", verbose=True):
	import numpy as np, time
	x = np.zeros((k,k), dtype=dtype)
	t =- time.time()
	x.dot(x)
	t += time.time()
	
	if verbose:
		print("NUMPY BLAS TEST:")
		print("GFLOP:    {:13.3f}".format(2*k**3/1e9))
		print("Time (s): {:16.6f}".format(t))
		print("GFLOPS:   {:13.3f}".format(2*k**3/1e9/t))
	
	return 2*k**3/t

