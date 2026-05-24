# -*- coding: utf-8 -*-
import argparse, ast, os, re, sys, warnings
from   argparse import (ArgumentParser,
                        Action,
                        Namespace,)
import nauka.utils.lr


#
# WARNING: EXTREMELY ugly string-parsing code.
# If it can be rewritten with re.match() magic, it should.
#
def _parseSpec(values):
	"""
	Split the argument string. Example:
	
	--arg name:value0,value1,key2=value2,key3=value3
	
	gets split into
	
	name, args, kwargs = (
	    "name",
	    (value0, value1),
	    {"key2":value2, "key3":value3},
	)
	"""
	split  = values.split(":", 1)
	name   = split[0].strip()
	rest   = split[1] if len(split) == 2 else ""
	args   = []
	kwargs = {}
	
	def carveRest(s, sep):
		quotepairs = {"'": "'", "\"": "\"", "{":"}", "[":"]", "(":")"}
		val   = ""
		quote = ""
		prevC = ""
		for i, c in enumerate(s):
			if quote:
				if   c == quote[-1]  and prevC != "\\":
					val    += c
					prevC   = ""
					quote   = quote[:-1]
				elif c in quotepairs and prevC != "\\":
					val    += c
					prevC   = ""
					quote  += quotepairs[c]
				elif prevC == "\\":
					val     = val[:-1]+c
					prevC   = ""
				else:
					val    += c
					prevC   = c
			else:
				if   c == sep:
					break
				elif c in quotepairs and prevC != "\\":
					val    += c
					prevC   = ""
					quote  += quotepairs[c]
				elif prevC == "\\":
					val     = val[:-1]+c
					prevC   = ""
				else:
					val    += c
					prevC   = c
			
		return val, s[i+1:]
	
	while rest:
		positionalVal, positionalRest = carveRest(rest, ",")
		keywordKey,    keywordRest    = carveRest(rest, "=")
		
		#
		# If the distance to the first "=" (or end-of-string) is STRICTLY
		# shorter than the distance to the first ",", we have found a
		# keyword argument.
		#
		
		if len(keywordKey)<len(positionalVal):
			key       = re.sub("\\s+", "", keywordKey)
			val, rest = carveRest(keywordRest, ",")
			try:    kwargs[key] = ast.literal_eval(val)
			except: kwargs[key] = val
		else:
			if len(kwargs) > 0:
				raise ValueError("Positional argument "+repr(r)+" found after first keyword argument!")
			val, rest = positionalVal, positionalRest
			try:    args += [ast.literal_eval(val)]
			except: args += [val]
	
	return name, args, kwargs


#
# Subcommand
#
class Subcommand         (object):
	cmdname       = None
	parserArgs    = {}
	subparserArgs = {}
	
	@classmethod
	def name(cls):
		return cls.cmdname or cls.__name__
	
	@classmethod
	def subcommands(cls):
		for s in cls.__dict__.values():
			if isinstance(s, type) and issubclass(s, Subcommand):
				yield s
	
	@classmethod
	def addToSubparser(cls, subp):
		argp = subp.add_parser(cls.name(), **cls.parserArgs)
		return cls.addAllArgs(argp)
	
	@classmethod
	def addArgs(cls, argp):
		pass
	
	@classmethod
	def addAllArgs(cls, argp=None):
		argp = argp or ArgumentParser(**cls.parserArgs)
		subc = list(cls.subcommands())
		if subc:
			subp = argp.add_subparsers(**cls.subparserArgs)
			for s in subc:
				s.addToSubparser(subp)
		argp.set_defaults(
		    __argp__ = argp,
		    __cls__  = cls,
		)
		cls.addArgs(argp)
		return argp
	
	@classmethod
	def run(cls, a, *args, **kwargs):
		a.__argp__.print_help()
		return 0

#
# Custom argument-parsing Actions
#
class Optimizer          (Action):
	def __init__(self, **kwargs):
		defaultOpt = kwargs.setdefault("default", "sgd")
		defaultOpt = Namespace(**self.parseOptSpec(defaultOpt))
		kwargs["default"] = defaultOpt
		kwargs["metavar"] = "OPTSPEC"
		kwargs["nargs"]   = None
		kwargs["type"]    = str
		kwargs.setdefault("help", "Optimizer selection.")
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		setattr(ns, self.dest, Namespace(**self.parseOptSpec(values)))
	
	def parseOptSpec   (cls, values):
		name, args, kwargs = _parseSpec(values)
		
		if   name in ["sgd"]:
			return cls.filterSGD(*args, **kwargs)
		elif name in ["nag"]:
			return cls.filterNAG(*args, **kwargs)
		elif name in ["adam"]:
			return cls.filterAdam(*args, **kwargs)
		elif name in ["amsgrad"]:
			return cls.filterAmsgrad(*args, **kwargs)
		elif name in ["rmsprop"]:
			return cls.filterRmsprop(*args, **kwargs)
		elif name in ["yf", "yfin", "yellowfin"]:
			return cls.filterYellowfin(*args, **kwargs)
		else:
			raise ValueError("Unknown optimizer \"{}\"!".format(name))
	
	@classmethod
	def fromFilter     (cls, d, name):
		d["name"] = name
		d.pop("cls", None)
		return d
	@classmethod
	def filterSGD      (cls, lr=1e-3, mom=0.9, nesterov=False):
		lr       = float(lr)
		mom      = float(mom)
		nesterov = bool(nesterov)
		
		assert lr    >  0
		assert mom   >= 0 and mom   < 1
		
		return cls.fromFilter(locals(), "sgd")
	@classmethod
	def filterNAG      (cls, lr=1e-3, mom=0.9, nesterov=True):
		return cls.fromFilter(cls.filterSGD(lr, mom, nesterov), "nag")
	@classmethod
	def filterAdam     (cls, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		lr    = float(lr)
		beta1 = float(beta1)
		beta2 = float(beta2)
		eps   = float(eps)
		
		assert lr    >  0
		assert beta1 >= 0 and beta1 < 1
		assert beta2 >= 0 and beta2 < 1
		assert eps   >= 0
		
		return cls.fromFilter(locals(), "adam")
	@classmethod
	def filterAmsgrad  (cls, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
		return cls.fromFilter(cls.filterAdam(lr, beta1, beta2, eps), "amsgrad")
	@classmethod
	def filterRmsprop  (cls, lr=1e-3, rho=0.9, eps=1e-8):
		lr    = float(lr)
		rho   = float(rho)
		eps   = float(eps)
		
		assert lr    >  0
		assert rho   >= 0 and rho   < 1
		assert eps   >= 0
		
		return cls.fromFilter(locals(), "rmsprop")
	@classmethod
	def filterYellowfin(cls, lr=1.0, mom=0.0, beta=0.999, curvWW=20, nesterov=False):
		lr       = float(lr)
		mom      = float(mom)
		beta     = float(beta)
		curvWW   = int(curvWW)
		nesterov = bool(nesterov)
		
		assert lr     >  0
		assert mom    >= 0 and mom  <  1
		assert beta   >= 0 and beta <= 1
		assert curvWW >= 3
		
		return cls.fromFilter(locals(), "yellowfin")

class CudaDevice         (Action):
	def __init__(self, **kwargs):
		#
		# If a CudaDevice flag is given, but without argument device IDs,
		# the default interpretation is "want CUDA, best guess at devices". The
		# default interpretation can be changed by providing the `const`
		# argument of argparse.Action.
		#
		if "const" not in kwargs:
			#
			# The default interpretation depends on the environment variable
			# CUDA_VISIBLE_DEVICES.
			#
			if "CUDA_VISIBLE_DEVICES" in os.environ:
				#
				# The environment has CUDA_VISIBLE_DEVICES set. We assume that
				# variable was wisely set, so we craft an ordered list of
				# integers as long as the variable indicates there are devices.
				#
				CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
				CUDA_VISIBLE_DEVICES = self.parseString(CUDA_VISIBLE_DEVICES)
				CUDA_VISIBLE_DEVICES = list(range(len(CUDA_VISIBLE_DEVICES)))
			else:
				#
				# The environment does not have CUDA_VISIBLE_DEVICES set. We'll
				# make a best guess that the user has and wants the first GPU.
				#
				CUDA_VISIBLE_DEVICES = [0]
		else:
			CUDA_VISIBLE_DEVICES = kwargs["const"]
		
		kwargs.setdefault("metavar", "DEVIDS")
		kwargs.setdefault("const",   CUDA_VISIBLE_DEVICES)
		kwargs["nargs"]   = "?"
		kwargs["default"] = []
		kwargs["type"]    = str
		kwargs.setdefault("help",    "List of CUDA device ids to use.")
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		#
		# If values is a list, it's probably because it's the const= list.
		# If values is a string, we parse it.
		#
		if isinstance(values, list):
			setattr(ns, self.dest, values)
		else:
			setattr(ns, self.dest, self.parseString(values))
	
	@classmethod
	def parseString(cls, dIdCommaSepString):
		#
		# Parse the device ID string.
		# 
		# We assume a comma-separated list of integers and hyphenated integer
		# ranges. Example:
		# 
		#           0,3-4,2 ,   1,   6-7
		#
		
		deviceIds = []
		
		for dIdRange in dIdCommaSepString.split(","):
			dIdRange      = dIdRange.strip()
			dIdRangeSplit = dIdRange.split("-")
			
			if   not dIdRange:
				continue
			elif len(dIdRangeSplit) == 0:
				pass
			elif len(dIdRangeSplit) == 1 and dIdRangeSplit[0]:
					deviceIds.append(int(dIdRangeSplit[0].strip()))
			elif len(dIdRangeSplit) == 2 and dIdRangeSplit[0] and dIdRangeSplit[1]:
				start = int(dIdRangeSplit[0].strip())
				end   = int(dIdRangeSplit[1].strip())
				step  = +1 if start <= end else -1
				deviceIds.extend(range(start, end+step, step))
			else:
				raise ValueError(
				    "Broken CUDA device range specification \"{}\"".format(dIdRange)
				)
		
		return deviceIds

class Preset             (Action):
	def __init__(self, **kwargs):
		if kwargs.get("default", argparse.SUPPRESS) != argparse.SUPPRESS:
			warnings.warn("A PresetAction ignores the default= keyword argument!")
		
		if "choices" not in kwargs:
			raise ValueError("Must provide a dictionary of presets!")
		else:
			self.presets = kwargs["choices"]
		if not isinstance(self.presets, dict):
			raise TypeError("Preset choices must be a dictionary!")
		
		for presetName, preset in self.presets.items():
			if not isinstance(presetName, str):
				raise TypeError("Preset names must be of type \"str\", but "
				                "one of them is not!")
			if not isinstance(preset,     list):
				raise TypeError("Presets must be a list of strings, but "
				                "one of them is not!")
		
		kwargs["default"] = argparse.SUPPRESS
		kwargs["choices"] = list(self.presets.keys())
		kwargs["nargs"]   = None
		kwargs["type"]    = str
		
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		parser.parse_args(self.presets[values], ns)

class FastDebug          (Action):
	def __init__(self, **kwargs):
		kwargs.setdefault("metavar", "N")
		kwargs.setdefault("const",   5)
		kwargs["const"]   = int(kwargs["const"])
		kwargs["nargs"]   = "?"
		kwargs["default"] = 0
		kwargs["type"]    = int
		kwargs.setdefault("help",
		    """
		    For debug purposes, run only a tiny number of epochs and iterations
		    per epoch (for both training & validation), thus exercising all of
		    the code quickly. The default is {:d} iterations.
		    """
		    [1:-1].format(kwargs["const"])
		)
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		setattr(ns, self.dest, values)

class _Dir               (Action):
	ENVVAR     = None
	DEFDEFVAL  = None
	HELPSTRING = None
	
	def __init__(self, **kwargs):
		"""
		For DirPathActions, the argument-handling logic is as follows:
		
		  - If arg explicitly given, use that.
		  - If arg not given but ENVVAR set, use that.
		  - If arg not given and ENVVAR unset, but default set, use that.
		  - If arg not given,    ENVVAR unset  and default unset, use DEFDEFVAL.
		"""
		defaultNoEnv = kwargs.get("default", self.DEFDEFVAL)
		default = os.environ.get(self.ENVVAR, defaultNoEnv)
		kwargs["default"] = default
		kwargs["nargs"]   = None
		kwargs.setdefault("type", str)
		kwargs.setdefault("help", self.HELPSTRING)
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		setattr(ns, self.dest, values)

class BaseDir            (_Dir):
	ENVVAR     = "NAUKA_BASEDIR"
	DEFDEFVAL  = "work"
	HELPSTRING = "Path to the base directory from which this experiment's true " \
	             "working directory will be derived."

class DataDir            (_Dir):
	ENVVAR     = "NAUKA_DATADIR"
	DEFDEFVAL  = "data"
	HELPSTRING = "Path to the datasets directory."

class TmpDir             (_Dir):
	ENVVAR     = "NAUKA_TMPDIR"
	DEFDEFVAL  = "tmp"
	HELPSTRING = "Path to a local, fast-storage, temporary directory."

class LRSchedule         (Action):
	def __init__(self, **kwargs):
		defaultOpt = kwargs.setdefault("default", [])
		if   isinstance(defaultOpt, (list, tuple, set)):
			defaultOpt = list(defaultOpt)
		elif isinstance(defaultOpt, (int, float)):
			defaultOpt = self.parseLRSpec("k:"+str(defaultOpt))
		elif isinstance(defaultOpt, str):
			defaultOpt = self.parseLRSpec(defaultOpt)
		else:
			raise ValueError("Invalid LR default="+repr(defaultOpt))
		kwargs["default"] = defaultOpt
		kwargs["metavar"] = "LRSPEC"
		kwargs["nargs"]   = None
		kwargs["type"]    = str
		kwargs.setdefault("help", "Learning rate schedule.")
		super().__init__(**kwargs)
	
	def __call__(self, parser, ns, values, option_string):
		setattr(ns, self.dest, self.parseLRSpec(values))
	
	@classmethod
	def parseLRSpec   (cls, values):
		values = values.split("*")
		for i in range(len(values)):
			try:
				values[i] = cls.filterConst   (float(values[i]))
			except:
				name, args, kwargs = _parseSpec(values[i])
				if   name in ["lambda", "prod", "product", "clamp"]:
					raise ValueError("LR schedule {} cannot be parsed from arguments!".format(repr(name)))
				elif name in ["k", "const", "constant"]:
					values[i] = cls.filterConst   (*args, **kwargs)
				elif name in ["step"]:
					values[i] = cls.filterStep    (*args, **kwargs)
				elif name in ["exp"]:
					values[i] = cls.filterExp     (*args, **kwargs)
				elif name in ["cos"]:
					values[i] = cls.filterCos     (*args, **kwargs)
				elif name in ["saw", "sawtooth"]:
					values[i] = cls.filterSawtooth(*args, **kwargs)
				elif name in ["tri", "triangle"]:
					values[i] = cls.filterTriangle(*args, **kwargs)
				elif name in ["plateau"]:
					values[i] = cls.filterPlateau (*args, **kwargs)
				else:
					raise ValueError("Unknown LR schedule {}!".format(repr(name)))
			
			values[i] = Namespace(**values[i])
		return values
	
	@classmethod
	def fromFilter    (cls, d, name):
		d["name"] = name
		d.pop("cls", None)
		return d
	@classmethod
	def filterConst   (cls, lr=1.0):
		lr    = float(lr)
		return cls.fromFilter(locals(), "const")
	@classmethod
	def filterStep    (cls, stepSize, gamma=0.95):
		stepSize = int(stepSize)
		gamma    = float(gamma)
		return cls.fromFilter(locals(), "step")
	@classmethod
	def filterExp     (cls, gamma=0.95):
		gamma    = float(gamma)
		return cls.fromFilter(locals(), "exp")
	@classmethod
	def filterCos     (cls, period, lrA=1.0, lrB=0.0):
		period = int(period)
		lrA    = float(lrA)
		lrB    = float(lrB)
		return cls.fromFilter(locals(), "cos")
	@classmethod
	def filterSawtooth(cls, period, lrA=1.0, lrB=0.0):
		period = int(period)
		lrA    = float(lrA)
		lrB    = float(lrB)
		return cls.fromFilter(locals(), "sawtooth")
	@classmethod
	def filterTriangle(cls, period, lrA=1.0, lrB=4.0):
		period = int(period)
		lrA    = float(lrA)
		lrB    = float(lrB)
		return cls.fromFilter(locals(), "triangle")
	@classmethod
	def filterPlateau (cls, patience=10, cooldown=0, gamma=0.1, threshold=1e-4,
	                   lrMin=0, eps=1e-8, mode="min", thresholdMode="rel",
	                   verbose=False):
		patience      = int(patience)
		cooldown      = int(cooldown)
		gamma         = float(gamma)
		threshold     = float(threshold)
		lrMin         = float(lrMin)
		eps           = float(eps)
		mode          = str(mode)
		thresholdMode = str(thresholdMode)
		verbose       = bool(verbose)
		return cls.fromFilter(locals(), "plateau")

