# -*- coding: utf-8 -*-

def getstate(password, salt="random", rounds=1, hash="sha256"):
	from ..numpy.random import get_state
	state = get_state(password, salt, rounds, hash)
	state = tuple(state[1].tolist()) + (624,)
	return (3, state, None)

def setstate(password, salt="random", rounds=1, hash="sha256"):
	import random
	state = getstate(password, salt, rounds, hash)
	random.setstate(state)
	return state

