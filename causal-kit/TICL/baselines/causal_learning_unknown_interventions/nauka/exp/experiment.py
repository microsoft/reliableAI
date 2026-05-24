#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, re


class Experiment(object):
	"""
	Experiment.
	
	An experiment comprises both an in-memory state and an on-disk state. At
	regular intervals, the in-memory state is synchronized with the on-disk
	state, thus permitting a resume should the experiment be killed. These
	on-disk serializations are called "snapshots".
	
	The hierarchical organization of files within an experiment is as follows:
	
		Experiment
		*   workDir/                   | The main working directory.
			->  snapshot/              | Snapshot directory
				-> <#>/                | Snapshot numbered <#>
					->  *.hdf5, ...    | Data files and suchlike.
				->  latest             | Symbolic link to latest snapshot.
	
	The invariants to be preserved are:
		- workDir/snapshot/latest either does not exist, or is a *symbolic link*
		  pointing to the directory `#`, which does exist and has a
		  complete, valid, loadable snapshot within it.
		- Snapshots are not modified in any way after they've been dumped,
		  except for deletions due to purging.
	"""
	
	def __init__(self, workDir="."):
		self.__workDir = os.path.abspath(workDir)
	
	
	#
	# Fundamental properties:
	#     workDir
	#     snapDir
	#     latestLink
	#     latestSnapshotNum
	#     nextSnapshotNum
	#
	@property
	def workDir(self): return self.__workDir
	@property
	def snapDir(self):
		return os.path.join(self.workDir, "snapshot")
	@property
	def latestLink(self):
		return os.path.join(self.snapDir, "latest")
	@property
	def latestSnapshotNum(self):
		if self.haveSnapshots:
			s = os.readlink(self.latestLink)
			s = int(s, base=10)
			assert(s >= 0)
			return s
		else:
			return -1
	@latestSnapshotNum.setter
	def latestSnapshotNum(self, n):
		assert isinstance(n, int)
		self.__markLatest(n)
	@property
	def nextSnapshotNum(self):
		return self.latestSnapshotNum+1
	@property
	def haveSnapshots(self):
		"""Check if we have at least one snapshot."""
		return os.path.islink(self.latestLink) and os.path.isdir(self.latestLink)
	
	#
	# Mutable State Management.
	#
	# To be implemented by user as he/she sees fit.
	#
	def load(self, path):
		"""
		Load state from given path.
		
		Restores the experiment to a state as close as possible to the one
		the experiment was in at the moment of the dump() that generated the
		checkpoint with the given `path`.
		
		Returns `self` afterwards.
		"""
		
		return self
	
	def dump(self, path):
		"""
		Dump state to the directory `path/`
		
		When invoked by the snapshot machinery, `path/` may be assumed to
		already exist. The state must be saved under that directory, but
		the contents of that directory and any hierarchy underneath it are
		completely freeform, except that the subdirectory `path/.experiment`
		must not be touched.
		
		When invoked by the snapshot machinery, the path's basename as given
		by os.path.basename(path) will be the number this snapshot will be
		be assigned, and it is equal to self.nextSnapshotNum.
		
		Returns `self`.
		"""
		
		return self
	
	def fromScratch(self):
		"""Start a fresh experiment, from scratch.
		
		Returns `self`."""
		
		assert(not os.path.lexists(self.latestLink) or
		           os.path.islink (self.latestLink))
		self.rmR(self.latestLink)
		return self
	
	def fromSnapshot(self, path):
		"""Start an experiment from a snapshot.
		
		Most likely, this method will invoke self.load(path) at an opportune
		time in its implementation.
		
		Returns `self`."""
		
		return self.load(path)
	
	
	#
	# High-level Snapshot & Rollback Management
	#
	def snapshot             (self):
		"""Take a snapshot of the experiment.
		
		Returns `self`."""
		nextSnapshotNum  = self.nextSnapshotNum
		nextSnapshotPath = self.getFullPathToSnapshot(nextSnapshotNum)
		
		if os.path.lexists(nextSnapshotPath):
			self.rmR(nextSnapshotPath)
		self.mkdirp(os.path.join(nextSnapshotPath, ".experiment"))
		return self.dump(nextSnapshotPath).__markLatest(nextSnapshotNum)
	
	def rollback             (self, n=None):
		"""Roll back the experiment to the given snapshot number.
		
		Returns `self`."""
		
		if n is None:
			if self.haveSnapshots: return self.fromSnapshot(self.latestLink)
			else:                  return self.fromScratch()
		elif isinstance(n, int):
			loadSnapshotPath = self.getFullPathToSnapshot(n)
			assert(os.path.isdir(loadSnapshotPath))
			return self.__markLatest(n).fromSnapshot(loadSnapshotPath)
		else:
			raise ValueError("n must be int, or None!")
	
	def purge                (self,
	                          strategy           = "klogn",
	                          keep               = None,
	                          deleteNonSnapshots = False,
	                          **kwargs):
		"""Purge snapshot directory of snapshots according to some strategy,
		preserving however a given "keep" list or set of snapshot numbers.
		
		Available strategies are:
		    "lastk":  Keep last k snapshots (Default: k=10)
		    "klogn":  Keep every snapshot in the last k, 2k snapshots in
		              the last k**2, 3k snapshots in the last k**3, ...
		              (Default: k=4. k must be > 1).
		
		Returns `self`."""
		
		assert(isinstance(keep, (list, set))  or  keep is None)
		keep = set(keep or [])
		if self.haveSnapshots:
			if   strategy == "lastk":
				keep.update(self.strategyLastK(self.latestSnapshotNum, **kwargs))
			elif strategy == "klogn":
				keep.update(self.strategyKLogN(self.latestSnapshotNum, **kwargs))
			else:
				raise ValueError("Unknown purge strategy "+str(None)+"!")
			keep.update(["latest", str(self.latestSnapshotNum)])
		keep = set(map(str, keep))
		
		snaps, nonSnaps    = self.listSnapshotDir(self.snapDir)
		dirEntriesToDelete = set()
		dirEntriesToDelete.update(snaps)
		dirEntriesToDelete.update(nonSnaps if deleteNonSnapshots else set())
		dirEntriesToDelete.difference_update(keep)
		for dirEntry in dirEntriesToDelete:
			self.rmR(os.path.join(self.snapDir, dirEntry))
		
		return self
	
	def getFullPathToSnapshot(self, n):
		"""Get the full path to snapshot n."""
		return os.path.join(self.snapDir, str(n))
	
	def __markLatest         (self, n):
		"""Atomically reroute the "latest" symlink in the working directory so
		that it points to the given snapshot number."""
		self.atomicSymlink(str(n), self.latestLink)
		return self
	
	#
	# Snapshot purge strategies.
	#
	@classmethod
	def strategyLastK(kls, n, k=10):
		"""Return the directory names to preserve under the LastK purge strategy."""
		return set(map(str, filter(lambda x:x>=0, range(n, n-k, -1))))
	
	@classmethod
	def strategyKLogN(kls, n, k=4):
		"""Return the directory names to preserve under the KLogN purge strategy."""
		assert(k>1)
		s = set([n])
		i = 0
		
		while k**i <= n:
			s.update(range(n, n-k*k**i, -k**i))
			i += 1
			n -= n % k**i
		
		return set(map(str, filter(lambda x:x>=0, s)))
	
	#
	# Filesystem Utilities
	#
	@classmethod
	def mkdirp(kls, path, mode=0o755):
		"""`mkdir -p path/to/folder`. Creates a folder and all parent
		directories if they don't already exist."""
		os.makedirs(path, mode=mode, exist_ok=True)
	
	@classmethod
	def isFilenameInteger(kls, name):
		"""Report whether the given string is a non-negative, 0-prefix-free,
		decimal integer."""
		return re.match("^(0|[123456789]\d*)$", name)
	
	@classmethod
	def listSnapshotDir(kls, path):
		"""Return the set of snapshot directories and non-snapshot directories
		under the given path."""
		snapshotSet    = set()
		nonsnapshotSet = set()
		try:
			entryList = os.listdir(path)
			for e in entryList:
				if kls.isFilenameInteger(e): snapshotSet   .add(e)
				else:                        nonsnapshotSet.add(e)
		except FileNotFoundError: pass
		finally:
			return snapshotSet, nonsnapshotSet
	
	@classmethod
	def rmR(kls, path):
		"""`rm -R path`. Deletes, but does not recurse into, symlinks.
		If the path does not exist, silently return."""
		if   os.path.islink(path) or os.path.isfile(path):
			os.unlink(path)
		elif os.path.isdir(path):
			walker = os.walk(path, topdown=False, followlinks=False)
			for dirpath, dirnames, filenames in walker:
				for f in filenames:
					os.unlink(os.path.join(dirpath, f))
				for d in dirnames:
					os.rmdir (os.path.join(dirpath, d))
			os.rmdir(path)
	
	@classmethod
	def atomicSymlink(kls, target, name):
		"""Same syntax as os.symlink, except that the new link called `name`
		will first be created with the `name` and `target`
		
		    `name.ATOMIC` -> `target`
		
		, then be atomically renamed to
		
		    `name` -> `target`
		
		, thus overwriting any previous symlink there. If a filesystem entity
		called `name.ATOMIC` already exists, it will be forcibly removed.
		"""
		
		linkAtomicName = name+".ATOMIC"
		linkFinalName  = name
		linkTarget     = target
		
		if os.path.lexists(linkAtomicName):
			kls.rmR(linkAtomicName)
		os.symlink(linkTarget,     linkAtomicName)
		
		################################################
		######## FILESYSTEM LINEARIZATION POINT ########
		######## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv ########
		os.rename (linkAtomicName, linkFinalName)
		######## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ########
		######## FILESYSTEM LINEARIZATION POINT ########
		################################################

