from dolfin import *
import numpy as np
set_log_active(False)

muvals = [1., 0.1, 0.01]
nvals = [8, 16, 32, 64]
for i in [1,2]:
	for err_type in ['H1', 'L2']:
		print('	    %s error for P%d elements\n' % (err_type, i))

		errs = np.zeros((5,4))

		print('           	  N = 8  	  N = 16 	  N = 32  	  N = 64')
		ncount = 0
		for N in nvals:
		
			mesh = UnitSquareMesh(N,N)
			V = FunctionSpace(mesh, 'Lagrange', i)
			V_1 = FunctionSpace(mesh, 'Lagrange', i+2)

			u = TrialFunction(V)
			v = TestFunction(V)

			bc0 = DirichletBC(V, Constant(0), 'x[0] < DOLFIN_EPS')
			bc1 = DirichletBC(V, Constant(1), 'x[0] > 1 - DOLFIN_EPS')
			bcs = [bc0, bc1]

			u_ = Function(V)

			f = Constant(0.0)
			
			mucount = 0
			for mu in muvals:
				a = mu*inner(grad(u),grad(v))*dx + u.dx(0)*v*dx
				L = f*v*dx

				solve(a == L, u_, bcs)
				
				u_ex = interpolate(Expression('(exp(x[0]/mu) - \
					1)/(exp(1./mu) - 1)', \
					mu=mu),V_1)

				errs[mucount][ncount] = (errornorm(u_ex,u_, err_type))

				mucount += 1
			ncount += 1

		for j in range(3):
			print('mu = ' + '{:.1e}'.format(muvals[j]) + '	' +  \
				'{:.3e}'.format(errs[j][0]) + \
				'	' + '{:.3e}'.format(errs[j][1]) + '	' + \
				'{:.3e}'.format(errs[j][2]) + \
				'	' + '{:.3e}'.format(errs[j][3]))
		print('\n')

