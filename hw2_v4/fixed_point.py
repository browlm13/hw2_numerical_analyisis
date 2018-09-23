#!/usr/bin/env python3

"""
	File name: fixed_point.py
	Python Version: 3.6

		Fixed point iteration solver.

	L.J. Brown
	Math5315 @ SMU
	Fall 2018
"""

__filename__ = "fixed_point.py"
__author__ = "L.J. Brown"

# internal libraries
import logging

# external libraries
import numpy as np

# initilize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fixed_point(Gfun, x, maxit, rtol, atol, output=True):
	for i in range(maxit):
		next_x = Gfun(x)

		if output:
			logger.info(" iter %3i, \t x_{n} = %g,\t||x_{n-1} - x_{n}|| = %g" % (i+1, next_x, abs(x - next_x)))
		if (abs(x - next_x) < atol + rtol*next_x): 
			return next_x
		x = next_x

	# log failure to converge
	logger.info("\n\nFailure to converge in fixed_point.\n\n")

	return x
	