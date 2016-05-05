from dolfin import *
import numpy as np
set_log_active(False)

txt = ['l2','linf','l1']
N = [100, 1000, 10000]
K = [1, 10, 100]
for i in range(len(txt)):
	for n in N:
		mesh = UnitIntervalMesh(n)
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
			nrm = norm(u_,txt[i])
			print(nrm)
			C = e/nrm*n
			print C
			print('\n')