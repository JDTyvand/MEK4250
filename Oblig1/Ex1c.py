from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
set_log_active(False)

nvals = [8,16,32,64]

for k in [1, 10, 100]:
	for err_type in ['H1', 'L2']:
		for i in [1,2]:
		
			x = []
			y = []
			Alist = []
			for N in nvals:

				mesh = UnitSquareMesh(N,N)

				h = 1.0/N

				V = FunctionSpace(mesh, 'Lagrange', i)
				V_1 = FunctionSpace(mesh, 'Lagrange', i+2)

				u = TrialFunction(V)
				v = TestFunction(V)

				bc0 = DirichletBC(V, Constant(0.0), 'x[0] < DOLFIN_EPS')
				bc1 = DirichletBC(V, Constant(0.0), 'x[0] > 1-DOLFIN_EPS')
				bcs = [bc0, bc1]


				f = Expression('pi*pi*(k*k + l*l)*sin(pi*k*x[0])*cos(pi*l*x[1])'\
					,k=k, l=k)

				a = inner(grad(u),grad(v))*dx
				L = f*v*dx

				u_ = Function(V)

				solve(a == L, u_, bcs)

				
				u_ex = interpolate(Expression('sin(pi*k*x[0])*cos(pi*l*x[1])', \
					k=k, l=k),V_1)

				A = errornorm(u_ex, u_, err_type, degree_rise=3)

				Alist.append(A)
				x.append(ln(h))
				y.append(ln(A))


			Alist = np.array(Alist)
			x = np.array(x)
			y = np.array(y)

			Arr = np.vstack([x, np.ones(len(x))]).T
			alpha, lnC = np.linalg.lstsq(Arr, y)[0]
			print('%s, k = %3d, P%d elements:\n' % (err_type, k, i))
			print('alpha/beta = %.3f, C = %9.3f\n' % (alpha, exp(lnC)))
			#print('alpha/beta = %.3f, C = %9.3f' % (alpha, lnC))

			counter = 0
			for N in nvals:
				h = 1./N
				RHS = exp(lnC)*h**alpha
				print('N = %d: Errornorm = %.5f, Ch^alpha/beta = %.5f' % (N, Alist[counter], RHS))
				if Alist[counter] < RHS:
					print('Error estimate is valid for N = %d!\n' % N)
				else:
					print('Error estimate is NOT valid for N = %d!\n' % N)
				counter += 1
			print('\n')
			"""
			plt.figure()
			plt.plot(x, y, 'o', label='Original data', markersize=10)
			plt.plot(x, alpha*x + lnC, 'r', label='Fitted line')
			plt.title('Fitted line for alpha for %s error with k = l = %d and P%d elements' \
				% (err_type, k, i))
			plt.xlabel(r'$\alpha$')
			plt.ylabel('ln(C)')
			plt.legend()
			plt.savefig('P' + str(i) + '_' + 'k' + str(k) + '_' + err_type + '.png')
			"""





