from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys
set_log_active(False)


lvals = [1., 100., 10000.]
nvals = [8,16,32,64]

print('        %d       %d     %d\n' % (lvals[0],lvals[1], lvals[2]))
for i in [0,1]:
	rvals = []
	for l in lvals:
		err = []
		hvals = []

		for N in nvals:

			mesh = UnitSquareMesh(N,N)

			h = 1.0/N

			hvals.append(h)

			V = VectorFunctionSpace(mesh, 'Lagrange', i+1)
			V_1 = VectorFunctionSpace(mesh, 'Lagrange', i+2)
			Q = FunctionSpace(mesh, 'Lagrange', 1)

			W = V*Q
			u, p = TrialFunctions(W)
			v, q = TestFunctions(W)

			def boundary(x, on_boundary): return on_boundary

			u_ex = interpolate(Expression(("pi*x[0]*cos(pi*x[0]*x[1])",\
					"-pi*x[1]*cos(pi*x[0]*x[1])")),V_1)
			
			bc = DirichletBC(W.sub(0), u_ex, boundary)
			bcs = [bc]

			mu = 1.

			f = interpolate(Expression(("mu * pow(pi,2) \
						 * (2 * x[1] * sin(pi*x[0]*x[1])\
						+ pi * x[0] *(pow(x[0],2) + \
						pow(x[1],2))*cos(pi*x[0]*x[1]))"
						,  "-mu * pow(pi,2) *\
						(2 * x[0] * sin(pi*x[0]*x[1]) + pi * x[1] \
						*(pow(x[0],2) + pow(x[1],2)) *\
						cos(pi*x[0]*x[1]))"),	mu=mu),V_1)

			F = mu*inner(grad(u),grad(v))*dx + inner(p, div(v))*dx \
			- dot(f,v)*dx + inner(div(u),q)*dx - (1./l)*p*q*dx
			up_ = Function(W)

			solve(lhs(F)==rhs(F), up_, bcs)

			u_, p_ = up_.split(True)	

			L2 = errornorm(u_, u_ex, degree_rise=1)
			err.append(L2)
		x = np.array(np.log(hvals))
		y = np.array(np.log(err))
		Arr = np.vstack([x, np.ones(len(x))]).T
		alpha, lnC = np.linalg.lstsq(Arr, y)[0]
		rvals.append(alpha)
	print('P%d   %.4f   %.4f   %.4f' % (i+1, rvals[0], rvals[1], rvals[2]))
