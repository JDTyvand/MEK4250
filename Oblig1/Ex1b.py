from dolfin import *
import numpy as np
set_log_active(False)

k_vals = [1, 10, 100]
l_vals = [1, 10, 100]


for h in [8, 16, 32, 64]:
	for i in [1,2]:
		mesh = UnitSquareMesh(h,h)
		print('h = %d, P%d elements:\n' % (h,i))
		V = FunctionSpace(mesh, 'Lagrange', i)
		V_1 = FunctionSpace(mesh, 'Lagrange', i+2)

		u = TrialFunction(V)
		v = TestFunction(V)

		bc0 = DirichletBC(V, Constant(0.0), 'x[0] < DOLFIN_EPS')
		bc1 = DirichletBC(V, Constant(0.0), 'x[0] > 1-DOLFIN_EPS')
		bcs = [bc0, bc1]


		L2_errs = np.zeros((3,3))
		H1_errs = np.zeros((3,3))
		for k in range(3):
			for l in range(3):

				f = Expression('pi*pi*(k*k + l*l)*sin(pi*k*x[0])*cos(pi*l*x[1])'\
					,k=k_vals[k], l=l_vals[l])

				a = inner(grad(u),grad(v))*dx
				L = f*v*dx

				u_ = Function(V)

				solve(a == L, u_, bcs)

				u_exp = Expression('sin(pi*k*x[0])*cos(pi*l*x[1])',\
					k=k_vals[k],l=l_vals[l])
				u_ex = interpolate(u_exp,V_1)

				L2_errs[k][l] = errornorm(u_ex, u_, 'l2', degree_rise=3)
				H1_errs[k][l] = errornorm(u_ex, u_, 'h1', degree_rise=3)
				
		print('L2 norm:\n')
		print('           	l = 1    	l = 10    	l = 100')
		for n in range(3):
			#print('k = %3d%-3.3f%-3.3f%-3.3f' % (k_vals[n], L2_errs[n][0], L2_errs[n][1], L2_errs[n][2]))
			print('k = ' + '{:4d}'.format(k_vals[n]) + '	' +  '{:f}'.format(L2_errs[n][0]) + '	' + '{:f}'.format(L2_errs[n][1]) + '	' + '{:f}'.format(L2_errs[n][2]))
		print('\n')

		print('H1 norm:\n')
		print('           	l = 1    	l = 10    	l = 100')
		for n in range(3):
			#print('k = %3d%-3.3f%-3.3f%-3.3f' % (k_vals[n], H1_errs[n][0], H1_errs[n][1], H1_errs[n][2]))
			print('k = ' + '{:4d}'.format(k_vals[n]) + '	' +  '{:f}'.format(H1_errs[n][0]) + '	' + '{:f}'.format(H1_errs[n][1]) + '	' + '{:f}'.format(H1_errs[n][2]))
		print('\n')




