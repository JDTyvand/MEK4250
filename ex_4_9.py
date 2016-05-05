from dolfin import *
import numpy as np
set_log_active(False)

N = [10]
K = [1, 10, 100]
for n in N:
	mesh = UnitCubeMesh(n,n,n)
	V = FunctionSpace(mesh, "CG", 1)
	V2 = FunctionSpace(mesh, "CG", 4)
	for k in K:

		def boundary(x, on_boundary):
			return on_boundary

		BC = DirichletBC(V, 0, boundary)
		BC2 = DirichletBC(V2, 0, boundary)
		
		u = TrialFunction(V)
		v = TestFunction(V)
		u_exp = Expression("sin(k*pi*x[0])", k=k)
		u_ex = project(u_exp, V2, BC2)

		f = Expression("k*k*pi*pi*sin(k*pi*x[0])", k=k)

		F = inner(grad(u), grad(v))*dx - f*v*dx

		u_ = Function(V)
		solve(lhs(F) == rhs(F), u_, BC)
		e = errornorm(u_ex,u_,"h1")
		print(e)
		h2norm = np.sqrt(0.5*(1 + (k*np.pi)**2 + (k*np.pi)**4))
		print(h2norm)
		C = e/h2norm*n
		print C
		print('\n')
