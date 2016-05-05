from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
set_log_active(False)

muvals = [1., 0.1, 0.01]
nvals = [8,16,32,64]

for mu in muvals:
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
			bc1 = DirichletBC(V, Constant(1.0), 'x[0] > 1-DOLFIN_EPS')
			bcs = [bc0, bc1]


			f = Constant(0.0)
			beta = h/2.

			v = v + beta*v.dx(0)

			a = mu*inner(grad(u),grad(v))*dx + u.dx(0)*v*dx
			L = f*v*dx

			u_ = Function(V)

			solve(a == L, u_, bcs)

			
			u_ex = interpolate(Expression('(exp(x[0]/mu) - \
				1)/(exp(1./mu) - 1)', \
				mu=mu),V_1)

			e = u_ex.dx(0) - u_.dx(0)
			e = project(e, V)

			A = np.sqrt(h*norm(e, 'l2')**2 + mu*norm(e, 'l2')**2)

			Alist.append(A)
			x.append(ln(h))
			y.append(ln(A))


		Alist = np.array(Alist)
		x = np.array(x)
		y = np.array(y)

		Arr = np.vstack([x, np.ones(len(x))]).T
		alpha, lnC = np.linalg.lstsq(Arr, y)[0]
		print('mu = %.3e, P%d elements:\n' % (mu, i))
		print('alpha/beta = %.3f, C = %9.3f\n' % (alpha, exp(lnC)))
		#print('alpha/beta = %.3f, C = %9.3f' % (alpha, lnC))

		counter = 0
		for N in nvals:
			h = 1./N
			RHS = exp(lnC)*h**alpha
			print('N = %d: Errornorm = %.3e, Ch^alpha/beta = %.3e' \
				% (N, Alist[counter], RHS))
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





