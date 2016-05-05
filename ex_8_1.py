from dolfin import *

N = 32

for deg in [1, 2, 3]:
	mesh = UnitSquareMesh(N,N)
	V = FunctionSpace(mesh, 'Lagrange', deg)

	u = TrialFunction(V)
	v = TestFunction(V)

	a = inner(grad(u),grad(v))*dx
	A = assemble(a)
	nonzeros = A.nnz()
	print nonzeros