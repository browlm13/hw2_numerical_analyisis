#!/usr/bin/env python3

"""
	File name: aitken_fp.py
	Python Version: 3.6

		Fixed point iteration solver using aitken acceleration.

	L.J. Brown
	Math5315 @ SMU
	Fall 2018
"""

__filename__ = "aitken_fp.py"
__author__ = "L.J. Brown"

# internal libraries
import logging

# external libraries
import numpy as np

# mylib libraries
from newton_interp import *
from newton2 import *

# initilize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aitken_fp(Gfun, x, maxit, rtol, atol, output=True):

	for i in range(maxit):
		x0 = x
		x1 = Gfun(x0)
		x2 = Gfun(x1)

		denominator = (x2 - 2*x1 + x0)

		if denominator != 0:
			r = (x2*x0 - x1*x1)/denominator
			s = r-x
			x = r
		else:
			x = x2

		if (abs(s) < atol + rtol*s):
			return x

	# log failure to converge
	logger.info("\n\nFailure to converge in aitken_fp.\n\n")

	return x


# testing
if "__main__" in __name__:

	SHOW_OUTPUT = True

	# log SHOW_OUTPUT value
	logger.info("\n\nSHOW_OUTPUT set to %s.\n\n" % SHOW_OUTPUT)

	# test functions:

	# use: 
	#    Gfun_a = test_functions['Gfun_a']
	#    Ffun_a = root_finding_conversion(Gfun_a)

	fixed_point_functions = {

		'Gfun_a' : lambda x: (x**2)/4 -x/2 -1,
		'Gfun_b' : lambda x: np.cos(x),
		'Gfun_c' : lambda x: (x/3 +x)/2,
		'Gfun_d' : lambda x: np.cosh(x)/x -np.arctan(x)
	}

	# intitial guesses:

	initial_guesses = {
		'Gfun_a' : 2,
		'Gfun_b' : 2,
		'Gfun_c' : 2,
		'Gfun_d' : 2
	}

	# For all problems use an absolute solution tolerance of 10−5, 
	# a relative solution tolerance of 10−10, 
	# allow a maximum of 100 iterations.
	maxit = 100
	atol = 10**(-5)
	rtol = 10**(-10)

	# interp_representation_error
	interp_representation_error = lambda Gfun, x_final: abs(Gfun(x_final) - x_final)

	# run trials
	for Gfun_name, Gfun in fixed_point_functions.items():

		Gfun = fixed_point_functions[Gfun_name]
		x = initial_guesses[Gfun_name]

		# find x*
		x_target = aitken_fp(Gfun, x, maxit, rtol, atol, output=SHOW_OUTPUT)

		# display interp_representation_error
		error = interp_representation_error(Gfun, x)
		logger.info("\n\nFinal fixed point target error using acceleration acceleration.\n \
			\n\tFunction name: %s, \n\n\t\t |Gfun(x) - x| = %s.\n" % (Gfun_name, error))


