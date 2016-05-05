from sympy import *

for k in range(2):
	x = symbols('x')
	udiff = diff(sin((k+1)*pi*x),x)
	I = udiff*udiff
	for i in range(k):
		udiff = diff(udiff, x)
		I += 
