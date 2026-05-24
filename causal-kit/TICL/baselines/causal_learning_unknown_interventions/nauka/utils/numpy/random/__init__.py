# -*- coding: utf-8 -*-

def get_state(password, salt="numpy.random", rounds=1, hash="sha256"):
	import nauka.utils, numpy as np
	uint32le = np.dtype(np.uint32).newbyteorder("<")
	buf      = nauka.utils.pbkdf2(624*4, password, salt, rounds, hash)
	buf      = np.frombuffer(buf, dtype=uint32le).copy("C")
	return ("MT19937", buf, 624, 0, 0.0)

def set_state(password, salt="numpy.random", rounds=1, hash="sha256"):
	import numpy as np
	npRandomState = get_state(password, salt, rounds, hash)
	np.random.set_state(npRandomState)
	return npRandomState

