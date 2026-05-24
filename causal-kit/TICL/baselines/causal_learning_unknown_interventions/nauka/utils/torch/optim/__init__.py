# -*- coding: utf-8 -*-

def fromSpec(params, spec, **kwargs):
	from torch.optim import SGD, RMSprop, Adam
	
	if   spec.name in ["sgd", "nag"]:
		return SGD    (params, spec.lr, spec.mom,
		               nesterov=(spec.name=="nag"),
		               **kwargs)
	elif spec.name in ["rmsprop"]:
		return RMSprop(params, spec.lr, spec.rho, spec.eps, **kwargs)
	elif spec.name in ["adam"]:
		return Adam   (params, spec.lr, (spec.beta1, spec.beta2), spec.eps,
		               **kwargs)
	elif spec.name in ["amsgrad"]:
		return Adam   (params, spec.lr, (spec.beta1, spec.beta2), spec.eps,
		               amsgrad=True,
		               **kwargs)
	else:
		raise NotImplementedError("Optimizer "+spec.name+" not implemented!")

def setLR(optimizer, lr):
	if   isinstance(lr, dict):
		for paramGroup in optimizer.param_groups:
			paramGroup["lr"] = lr[paramGroup]
	elif isinstance(lr, list):
		for paramGroup, lritem in zip(optimizer.param_groups, lr):
			paramGroup["lr"] = lritem
	else:
		for paramGroup in optimizer.param_groups:
			paramGroup["lr"] = lr
