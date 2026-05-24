# -*- coding: utf-8 -*-
import math, os, time, contextlib


def iso8601timestamp(T=None, nanos=True, utc=False):
	"""Get ISO8601-formatted timestamp string."""
	T  = time.time() if T is None else T
	Ti = math.floor(T)
	Tn = round((T-Ti)*1e9)
	if Tn >= 1e9:
		Ti += 1
		Tn  = 0
	
	s  = time.gmtime(Ti)      if utc   else time.localtime(Ti)
	f  = time.strftime("%Y%m%dT%H%M%S", s)
	n  = ".{:09d}".format(Tn) if nanos else ""
	tz = "Z"                  if utc   else time.strftime("%z", s)
	return f+n+tz


def createWorkDir(baseDir,
                  projName,
                  expUUID,
                  expNames = [],
                  nanos    = True,
                  utc      = False):
	"""Create working directory for experiment if not existing already."""
	
	#
	# First, ensure the project's top-level hierarchy, especially by-uuid/,
	# exists, so that the only possible failure is due to the creation of
	# one additional directory.
	#
	projDir    = os.path.join(baseDir, projName)
	byuuidDir  = os.path.join(projDir, "by-uuid")
	bytimeDir  = os.path.join(projDir, "by-time")
	bynameDir  = os.path.join(projDir, "by-name", *expNames)
	byuuidPath = os.path.join(byuuidDir, expUUID)
	os.makedirs(byuuidDir, mode=0o755, exist_ok=True)
	os.makedirs(bytimeDir, mode=0o755, exist_ok=True)
	os.makedirs(bynameDir, mode=0o755, exist_ok=True)
	
	#
	# Attempt the creation of the experiment workDir by its UUID. Record
	# whether we were the original creators.
	#
	try:
		preexisting = False
		os.makedirs(byuuidPath,
		            mode     = 0o755,
		            exist_ok = False)
	except FileExistsError:
		preexisting = True
	
	#
	# If we were the first to create this working directory, additionally
	# make symlinks pointing to it from the auxiliary directories.
	#
	if not preexisting:
		expTime     = iso8601timestamp(nanos=nanos, utc=utc)
		expTimeUUID = expTime+"-"+expUUID
		bytimePath  = os.path.join(bytimeDir, expTimeUUID)
		bynamePath  = os.path.join(bynameDir, expUUID)
		os.symlink(os.path.relpath(byuuidPath, bytimeDir), bytimePath, True)
		os.symlink(os.path.relpath(byuuidPath, bynameDir), bynamePath, True)
	
	#
	# Return the constructed workDir.
	#
	return byuuidPath
