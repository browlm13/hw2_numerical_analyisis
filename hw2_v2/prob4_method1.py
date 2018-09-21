#!/usr/bin/env python3

"""
	File name: prob4_method1.py
	Python Version: 3.6

		Fixed point iteration solver.

		Method:

			1.) Convert function g(x) to root finding form. 
				Where f(x*) = 0 is the root finding form of the fixed point problem function g(x*) = x(*).

									f(x*) = g(x*) - x* = 0.

			2.) Compute the coefficients of the newton interpolating polynomial of f(x)
				so the derivative, f'(x), is easy to compute.

									y = Ffun(x) - where x is selected nodes
									c = newton_interp.coeffients(x,y)
									c_prime = np.polyder(c)
									Jfun = lambda xi: np.polyval(c_prime, xi)

					Note: f(x) = Ffun(x), f'(x) = Jfun(x) in order to match newtons method syntax.

			3.) Use Ffun, Jfun and newtons method to find x*, which hopefully is close to Gfun(x*).

	L.J. Brown
	Math5315 @ SMU
	Fall 2018
"""

__filename__ = "prob4_method1.py"
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

def root_finding_form(Gfun, n=10):
	"""
		Convert the fixed point problem function to a form for previously defined newton method, method
		returns Ffun, Jfun for use in newtons method:

			use:
				Ffun, Jfun = root_finding_form(Gfun)
				x_traget = newton(Ffun, Jfun, x, maxit, Srtol, Satol, Rrtol, Ratol, output)

		1.) Convert function g(x) to root finding form, 'f(x*) = g(x*) - x* = 0', to find Ffun for newtons method.
		2.) Constructing a newton interpolating polynomial and find its derivative to find Jfun for newtons method.

		:param Gfun: fixed point problem function.
		:param n: 'int' of number of nodes given when constructing newton interpolating polynomial, defualt 10.
		:returns: Ffun, Jfun to be used as parameters in newtons method defined in the file 'newton.py'.
	"""

	#
	# 	Convert to root finding problem
	#

	# function to convert to root finding problem given g(x). 'g(x*) = x*' -> 'f(x*) = 0'
	root_finding_conversion = lambda Gfun: lambda x: Gfun(x) -x 

	# convert
	Ffun = root_finding_conversion(Gfun)

	#
	# 	Select nodes for constructing newton interpolating polynomial. 
	#
	# 		Note: Currently linearly spaced evalutations of f(x).
	# 		TODO: Use chebyshev nodes.
	#

	# compute x and y data points
	x = np.linspace(-1,1,n)
	y = Ffun(x)

	# compute coefficients of interpolating polynomial
	c = coeffients(x,y)

	# compute coefficients for first derivative of p with coefficients c
	c_prime = np.polyder(c)

	# construct Jfun lambda function for derivative of interpolating polynomial
	# to use in newtons method
	Jfun = lambda xi: np.polyval(c_prime, xi)

	return Ffun, Jfun

# testing
if "__main__" in __name__:

	SHOW_OUTPUT = False

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

		# convert Gfun to parameters useful in newtons method newton, 
		#    newton(Ffun, Jfun, x, maxit, rtol, atol, output)
		Ffun, Jfun = root_finding_form(Gfun)

		# find x*
		x_target = newton2(Ffun, Jfun, x, maxit, rtol, atol, output=SHOW_OUTPUT)

		# display interp_representation_error
		error = interp_representation_error(Gfun, x)
		logger.info("\n\nFinal fixed point target error using newton interpolating polynomial in place of Ffun.\n \
			\n\tFunction name: %s, \n\n\t\t |Gfun(x) - x| = %s.\n" % (Gfun_name, error))


	
