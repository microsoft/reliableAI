# -*- coding: utf-8 -*-
import numbers, math



#
# Parse from spec
#
def fromSpec(spec):
	if   isinstance(spec, numbers.Real):
		return ConstLR(spec)
	elif spec.name == "const":
		return ConstLR(spec.lr)
	elif spec.name == "step":
		return StepLR(spec.stepSize, spec.gamma)
	elif spec.name == "exp":
		return ExpLR(spec.gamma)
	elif spec.name == "cos":
		return CosLR(spec.period, spec.lrA, spec.lrB)
	elif spec.name == "sawtooth":
		return SawtoothLR(spec.period, spec.lrA, spec.lrB)
	elif spec.name == "triangle":
		return TriangleLR(spec.period, spec.lrA, spec.lrB)
	elif spec.name == "plateau":
		return PlateauLR(spec.patience,  spec.cooldown,      spec.gamma,
		                 spec.threshold, spec.lrMin,         spec.eps,
		                 spec.mode,      spec.thresholdMode, spec.verbose)
	else:
		raise NotImplementedError("LR Schedule "+spec.name+" not implemented!")

def fromSpecList(specs):
	if   len(specs) == 0:
		return ConstLR()
	elif len(specs) == 1:
		return fromSpec(*specs)
	else:
		return ProdLR(*[fromSpec(s) for s in specs])


#
# LR Schedules
#

class _LRBase(numbers.Real):
	__slots__ = ["__dict__", "__weakref__", "_stepNum"]
	def __new__      (cls, *args, **kwargs):
		lr = super().__new__(cls)
		lr._stepNum = 0
		return lr
	def __bool__     (self):    return bool(float(self))
	def __abs__      (self):    return abs(float(self))
	def __add__      (self, x): return float(self) +  x
	def __ceil__     (self):    return math.ceil(float(self))
	def __divmod__   (self, x): return divmod(float(self), x)
	def __eq__       (self, x): return float(self) == x
	def __floor__    (self):    return math.floor(float(self))
	def __floordiv__ (self, x): return float(self) // x
	def __format__   (self, f): return format(float(self), f)
	def __ge__       (self, x): return float(self) >= x
	def __gt__       (self, x): return float(self) >  x
	def __le__       (self, x): return float(self) <= x
	def __lt__       (self, x): return float(self) <  x
	def __mod__      (self, x): return float(self) %  x
	def __mul__      (self, x): return float(self) *  x
	def __neg__      (self):    return -float(self)
	def __pos__      (self):    return +float(self)
	def __pow__      (self, x): return pow(float(self), x)
	def __radd__     (self, x): return x +  float(self)
	def __rdivmod__  (self, x): return divmod(x, float(self))
	def __repr__     (self):    return "_LRBase()"
	def __rfloordiv__(self, x): return x // float(self)
	def __rmod__     (self, x): return x %  float(self)
	def __rmul__     (self, x): return x *  float(self)
	def __round__    (self, n): return round(float(self), n)
	def __rpow__     (self, x): return pow(x, float(self))
	def __rsub__     (self, x): return x -  float(self)
	def __rtruediv__ (self, x): return x /  float(self)
	def __str__      (self):    return repr(self) + " @ "+str(self.stepNum)
	def __sub__      (self, x): return float(self) -  x
	def __truediv__  (self, x): return float(self) /  x
	def __trunc__    (self):    return math.trunc(float(self))
	
	def step         (self, *, memo=None, **kwargs):
		if memo is None:
			memo = set() # Recursive descent begins now.
		if id(self) not in memo:
			memo.add(id(self))
			self._step(**kwargs)
			for child in self.children:
				if isinstance(child, _LRBase):
					child.step(memo=memo, **kwargs)
		return self
	
	def _step        (self, *args, **kwargs):
		self._stepNum += 1
	
	@property
	def stepNum      (self):    return int(self._stepNum)
	@property
	def children     (self):    return []

class ConstLR(_LRBase):
	__slots__ = ["_lr"]
	def __init__     (self, lr=1.0):
		self._lr = float(lr)
	def __float__    (self):
		return self._lr
	def __repr__     (self):
		return "ConstLR(lr={})".format(repr(self._lr))


class LambdaLR(_LRBase):
	__slots__ = ["_lrLambda"]
	def __init__     (self, lrLambda=lambda stepNum: 1.0):
		self._lrLambda = lrLambda
	def __float__    (self):
		return self._lrLambda(self.stepNum)
	def __repr__     (self):
		return "LambdaLR(lrLambda={})".format(repr(self._lrLambda))


class ProdLR(_LRBase):
	__slots__ = ["_children"]
	def __init__     (self, *children):
		self._children = children
	def __float__    (self):
		product = 1.0
		for lr in self._children: product *= float(lr)
		return product
	def __repr__     (self):
		return "ProdLR({})".format(", ".join(
			[repr(child) for child in self._children]
		))
	@property
	def children     (self): return self._children


class ClampLR(_LRBase):
	__slots__ = ["_lr", "_lrMin", "_lrMax"]
	def __new__      (cls,  lr, lrMin=None, lrMax=None):
		if lr is None:
			raise ValueError("Invalid LR: "+repr(lr))
		if lrMin is None and lrMax is None:
			# Don't bother creating a wrapper ClampLR object if it isn't going
			# to accomplish anything whatsoever.
			if   isinstance(lr, _LRBase):
				return lr
			else:
				try:
					return ConstLR(lr)
				except e:
					raise ValueError("Invalid LR: "+repr(lr)) from e
		return super().__new__(cls)
	def __init__     (self, lr, lrMin=None, lrMax=None):
		self._lr    = lr
		self._lrMin = lrMin if lrMin is None else float(lrMin)
		self._lrMax = lrMax if lrMax is None else float(lrMax)
	def __float__    (self):
		lr = float(self._lr)
		lr = lr if self._lrMax is None else min(lr, self._lrMax)
		lr = lr if self._lrMin is None else max(lr, self._lrMin)
		return lr
	def __repr__     (self):
		return "ClampLR(lr={}, lrMin={}, lrMax={})".format(
		    repr(self._lr),
		    repr(self._lrMin),
		    repr(self._lrMax),
		)
	@property
	def children     (self): return [self._lr]


class StepLR(_LRBase):
	__slots__ = ["_stepSize", "_gamma"]
	def __init__     (self, stepSize, gamma=0.95):
		self._stepSize = int(stepSize)
		self._gamma    = float(gamma)
	def __float__    (self):
		return self._gamma ** (self.stepNum // self._stepSize)
	def __repr__     (self):
		return "StepLR(stepSize={}, gamma={})".format(
		    repr(self._stepSize),
		    repr(self._gamma)
		)


class ExpLR(_LRBase):
	__slots__ = ["_gamma"]
	def __init__     (self, gamma=0.95):
		self._gamma = float(gamma)
	def __float__    (self):
		return self._gamma ** self.stepNum
	def __repr__     (self):
		return "ExpLR(gamma={})".format(repr(self._gamma))


class CosLR(_LRBase):
	__slots__ = ["_period", "_lrA", "_lrB"]
	def __init__     (self, period, lrA=1.0, lrB=0.0):
		self._period = int(period)
		self._lrA    = float(lrA)
		self._lrB    = float(lrB)
	def __float__    (self):
		if self._period < 2: return self._lrA
		periodf = self._period
		modstep = self.stepNum % periodf
		cosine  = math.cos(2*math.pi*modstep/periodf)
		factor  = 0.5*(1.0-cosine)
		return self._lrA + (self._lrB-self._lrA)*factor
	def __repr__     (self):
		return "CosLR(period={}, lrA={}, lrB={})".format(
		    repr(self._period),
		    repr(self._lrA),
		    repr(self._lrB),
		)


class SawtoothLR(_LRBase):
	__slots__ = ["_period", "_lrA", "_lrB"]
	def __init__     (self, period, lrA=1.0, lrB=0.0):
		self._period = int(period)
		self._lrA    = float(lrA)
		self._lrB    = float(lrB)
	def __float__    (self):
		if self._period < 2: return self._lrA
		periodf   = self._period
		modstep   = self.stepNum % periodf
		factor    = modstep/(periodf-1.0)
		return self._lrA + (self._lrB-self._lrA)*factor
	def __repr__     (self):
		return "SawtoothLR(period={}, lrA={}, lrB={})".format(
		    repr(self._period),
		    repr(self._lrA),
		    repr(self._lrB),
		)


class TriangleLR(_LRBase):
	__slots__ = ["_period", "_lrA", "_lrB"]
	def __init__     (self, period, lrA=1.0, lrB=4.0):
		self._period = int(period)
		self._lrA    = float(lrA)
		self._lrB    = float(lrB)
	def __float__    (self):
		if self._period < 2: return self._lrA
		periodf   = self._period
		modstep   = self.stepNum % periodf
		period1   = periodf//2
		period2   = periodf-period1
		firsthalf = modstep < period1
		periodh   = period1 if firsthalf else period2
		hmodstep  = modstep if firsthalf else periodf-modstep
		factor    = hmodstep/periodh
		return self._lrA + (self._lrB-self._lrA)*factor
	def __repr__     (self):
		return "TriangleLR(period={}, lrA={}, lrB={})".format(
		    repr(self._period),
		    repr(self._lrA),
		    repr(self._lrB),
		)


class PlateauLR(_LRBase):
	"""Mostly lifted from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau"""
	__slots__ = ["_patience", "_cooldown", "_gamma", "_threshold", "_lrMin",
	             "_eps", "_mode", "_thresholdMode", "_verbose", "_lr", "_best",
	             "_numBadSteps", "_cooldownCounter"]
	def __init__     (self, patience=10, cooldown=0, gamma=0.1, threshold=1e-4,
	                        lrMin=0, eps=1e-8, mode="min", thresholdMode="rel",
	                        verbose=False):
		gamma         = float(gamma)
		mode          = str(mode)
		thresholdMode = str(thresholdMode)
		if gamma >= 1.0:
			raise ValueError("Invalid gamma {} >= 1.0!".format(repr(gamma)))
		if mode          not in {"min", "max"}:
			raise ValueError("Unknown mode={}!".format(repr(mode)))
		if thresholdMode not in {"rel", "abs"}:
			raise ValueError("Unknown thresholdMode={}!".format(repr(thresholdMode)))
		
		self._patience        = int  (patience)
		self._cooldown        = int  (cooldown)
		self._gamma           = float(gamma)
		self._threshold       = float(threshold)
		self._lrMin           = float(lrMin)
		self._eps             = float(eps)
		self._mode            = str  (mode)
		self._thresholdMode   = str  (thresholdMode)
		self._verbose         = bool (verbose)
		
		self._lr              = 1.0
		self._best            = math.inf if mode=="min" else -math.inf
		self._numBadSteps     = 0
		self._cooldownCounter = 0
	
	def __float__    (self):
		return self._lr
	
	def __repr__     (self):
		return "PlateauLR(patience={}, cooldown={}, gamma={}, threshold={}, " \
		       "lrMin={}, eps={}, mode={}, thresholdMode={}, verbose={})".format(
		    repr(self._patience),
		    repr(self._cooldown),
		    repr(self._gamma),
		    repr(self._threshold),
		    repr(self._lrMin),
		    repr(self._eps),
		    repr(self._mode),
		    repr(self._thresholdMode),
		    repr(self._verbose),
		)
	
	def _step        (self, metric=None):
		if metric is None:
			return
		metric = float(metric)
		
		if self._isNewBest(metric):
			self._best         = metric
			self._numBadSteps  = 0
		else:
			self._numBadSteps += 1
		
		if self._coolingDown:
			self._cooldownCounter -= 1
			self._numBadSteps      = 0
		
		if self._numBadSteps > self._patience:
			self._cooldownCounter = self._cooldown
			self._numBadSteps     = 0
			lrOld = float(self._lr)
			lrNew = max(lrOld*self._gamma, self._lrMin)
			if lrOld-lrNew > self._eps:
				self._lr = lrNew
				if self._verbose:
					print("Step {:6d}: Reducing LR from {:.4e} to {:.4e} .".format(
					      self.stepNum, lrOld, lrNew
					))
		
		super()._step()
	
	def _isNewBest(self, metric):
		if self._mode=="min":
			if self._thresholdMode=="rel":
				dynamicBest = self._best*(1.-self._threshold)
			else:
				dynamicBest = self._best-self._threshold
			return metric < dynamicBest
		else:
			if self._thresholdMode=="rel":
				dynamicBest = self._best*(1.+self._threshold)
			else:
				dynamicBest = self._best+self._threshold
			return metric > dynamicBest
	
	@property
	def _coolingDown (self):
		return self._cooldownCounter > 0
	
