from dolfin import *

for n in [100, 1000, 10000]:
	mesh = UnitCubeMesh(n,n,n)
	V = FunctionSpace(mesh, "Lagrange", 1)
	W = FunctionSpace(mesh, "Lagrange", 5)
	u = TestFunction(V)
	v = TrialFunction(V)
	for k in [1, 10, 100]:
		u_exp = Expression("sin(k*pi*x[0])", k=k)
		u_ex = project(u_exp, W)

		f = Expression("k*k*pi*pi*sin(k*pi*x[0])", k=k)

		F = inner(grad(u), grad(v))*dx - f*v*dx

		u_ = Function(V)
		solve(lhs(F) == rhs(F), u_)
		H1 = errornorm(u_ex,u_,"h1")
		print(H1)
		"""
		L2 = norm(u_ex)
		C = n*L2/H1
		print C
		"""