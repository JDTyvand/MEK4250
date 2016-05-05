from dolfin import *
import sys
set_log_active(False)


lvals = [1., 100., 10000.]
nvals = [8,16,32,64]

for i in [0,1]:
	print("P%d Elements\n" % (i+1))
	for l in lvals:
		for N in nvals:

			mesh = UnitSquareMesh(N,N)

			h = 1.0/N

			V = VectorFunctionSpace(mesh, 'Lagrange', i+1)
			V_1 = VectorFunctionSpace(mesh, 'Lagrange', i+1)

			u = TrialFunction(V)
			v = TestFunction(V)


			u_ex = interpolate(Expression(("pi*x[0]*cos(pi*x[0]*x[1])",\
					"-pi*x[1]*cos(pi*x[0]*x[1])")),V_1)
			
			bc = DirichletBC(V, u_ex, "on_boundary")
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
			

			a = mu*inner(grad(u),grad(v))*dx + l*inner(div(u),div(v))*dx
			L =	dot(f,v)*dx
			u_ = Function(V)

			solve(a == L, u_, bcs)	
			
			err = errornorm(u_, u_ex, norm_type='l2', degree_rise=1)
			print ('h = %f   lambda = %-5d  error = %e' % (h, l, err))
		print
	print
