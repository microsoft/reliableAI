# -*- coding: utf-8 -*-

def manual_seed    (password, salt="torch.cuda", rounds=1, hash="sha256"):
	import nauka.utils.torch.random, torch
	seed = nauka.utils.torch.random.make_seed(password, salt, rounds, hash)
	torch.cuda.manual_seed(seed)
	return seed

def manual_seed_all(password, salt="torch.cuda", rounds=1, hash="sha256"):
	import nauka.utils.torch.random, torch
	seed = nauka.utils.torch.random.make_seed(password, salt, rounds, hash)
	torch.cuda.manual_seed_all(seed)
	return seed

#
# It would be nice to implement
#   - get_rng_state()
#   - get_rng_state_all()
#   - set_rng_state()
#   - set_rng_state_all()
# , but the PyTorch ByteTensor format for the MT RNG is much more unstable and
# undocumented than Numpy's. For now we are forced to proceed through the
# entropic chokepoint of PyTorch's 64-bit seed, instead of (as in Numpy)
# directly generating the 2.5KB state with PBKDF2.
#
