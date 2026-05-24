# -*- coding: utf-8 -*-

#
# Random assorted stuff.
#
# Functions and utilities in this module may ONLY rely on the presence of
# standard Python library packages, plus numpy.
#

from   .  import lr, numpy, random, torch


def toBytesUTF8(x, errors="strict"):
	return x.encode("utf-8", errors=errors) if isinstance(x, str) else x

def pbkdf2                   (dkLen, password, salt="", rounds=1, hash="sha256"):
	import hashlib
	password = toBytesUTF8(password)
	salt     = toBytesUTF8(salt)
	return hashlib.pbkdf2_hmac(hash, password, salt, rounds, dkLen)

def pbkdf2int                (nbits, password, salt="", rounds=1, hash="sha256", signed=False):
	nbits = int(nbits)
	dkLen = (nbits+7) // 8
	ebits =  nbits     % 8
	ebits =  8-ebits if ebits else 0
	buf   = pbkdf2(dkLen, password, salt, rounds, hash)
	i     = int.from_bytes(buf, "little", signed=signed) >> ebits
	return i

class PlainObject(object):
	pass
