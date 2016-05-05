from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys
set_log_active(False)

plt.figure(1)
plt.figure(2)
for i in [[4,3], [4,2], [3,2], [3,1]]:
	print('P%d-P%d' %(i[0],i[1]))
	hvals = []
	UE = []
	PE = []
	for N in [2, 4, 8, 16, 32, 64]:
		h = 1./N
		hvals.append(h)
		mesh = UnitSquareMesh(N,N)
		V = VectorFunctionSpace(mesh, 'Lagrange', i[0])
		Q = FunctionSpace(mesh, 'Lagrange', i[1])

		W = MixedFunctionSpace([V, Q])

		u, p = TrialFunctions(W)
		v, q = TestFunctions(W)

		f = Expression(("pow(pi,2)*sin(pi*x[1]) - 2*pi*cos(2*pi*x[0])",\
				 		"pow(pi,2)*cos(pi*x[0])"))

		u_ex = Expression(("sin(pi*x[1])","cos(pi*x[0])"))

		p_ex = Expression("sin(2*pi*x[0])")

		bc_u = DirichletBC(W.sub(0), u_ex, "on_boundary")
		bc_p = DirichletBC(W.sub(1), p_ex, "on_boundary")
		bc = [bc_u, bc_p]

		a = inner(grad(u), grad(v))*dx + div(u)*q*dx + div(v)*p*dx
		L = inner(f, v)*dx

		up_ = Function(W)
		A, b = assemble_system(a, L, bc)
		solve(a == L, up_, bc)

		u_, p_ = up_.split(True)

		Uerr = errornorm(u_ex,u_, norm_type='h1', degree_rise=1)
		UE.append(Uerr)
		Perr = errornorm(p_ex, p_, norm_type='l2', degree_rise=1)
		PE.append(Perr)


	Ur = []
	Pr = []
	for j in range(1,len(UE)):
		Uconv = np.log(UE[j-1]/UE[j])/np.log(hvals[j-1]/hvals[j])
		Ur.append(Uconv)
		Pconv = np.log(PE[j-1]/PE[j])/np.log(hvals[j-1]/hvals[j])
		Pr.append(Pconv)
		print ('h = %f    r for U = %f    r for P = %f' % (hvals[j], Uconv, Pconv))
	plt.figure(1)
	plt.loglog(hvals,UE, label='P%d-P%d' % ((i[0],i[1])))
	plt.figure(2)
	plt.loglog(hvals,PE, label='P%d-P%d' % ((i[0],i[1])))

	
plt.figure(1)
plt.legend(loc='upper left')
plt.savefig('Velocity_convergence.png')

plt.figure(2)
plt.legend(loc='upper left')
plt.savefig('Pressure_convergence.png')
