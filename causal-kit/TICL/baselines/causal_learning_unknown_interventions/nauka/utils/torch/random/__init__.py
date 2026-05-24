# -*- coding: utf-8 -*-

def make_seed  (password, salt="torch.random", rounds=1, hash="sha256"):
	from nauka.utils import pbkdf2int
	return pbkdf2int(64, password, salt, rounds, hash, signed=True)

def manual_seed(password, salt="torch.random", rounds=1, hash="sha256"):
	import torch
	seed = make_seed(password, salt, rounds, hash)
	torch.random.manual_seed(seed)
	return seed


#
# It would be nice to implement
#   - get_rng_state()
#   - set_rng_state()
# , but the PyTorch ByteTensor format for the MT RNG is much more unstable and
# undocumented than Numpy's. For now we are forced to proceed through the
# entropic chokepoint of PyTorch's 64-bit seed, instead of (as in Numpy)
# directly generating the 2.5KB state with PBKDF2.
#
