from dolfin import *
import numpy as np
import matplotlib.pyplot as 

Ns = [10,20,40,80]

for N in Ns:

	mesh = UnitIntervalMesh(N)

	V = FunctionSpace(V, "Lagrange", 1)

	u = TrialFunction(V)
	v = TestFunction(V)

	