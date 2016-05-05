from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys
set_log_active(False)
plt.figure(1)
plt.figure(2)

mu = 1.

def sigma(u,p):
	return 2.*mu*sym(grad(u))

def eps(u):
	return sym(u)

class wallBoundary(SubDomain):
	def inside(self, x, on_boundary): 
			return x[1] < DOLFIN_EPS or x[1] > 1-DOLFIN_EPS
plt.figure()
for i in [[4,3], [4,2], [3,2], [3,1]]:
	print('P%d-P%d' % (i[0],i[1]))
	hvals = []
	SE = []
	for N in [2, 4, 8, 16, 32, 64]:
		h = 1./N
		hvals.append(h)
		mesh = UnitSquareMesh(N,N)
		V = VectorFunctionSpace(mesh, 'Lagrange', i[0])
		V_1 = VectorFunctionSpace(mesh, 'Lagrange', i[0]+1)
		Q = FunctionSpace(mesh, 'Lagrange', i[1])
		Q_1 = FunctionSpace(mesh, 'Lagrange', i[1]+1)

		W = MixedFunctionSpace([V, Q])

		u, p = TrialFunctions(W)
		v, q = TestFunctions(W)

		walls = wallBoundary()
		boundaries = FacetFunction("size_t", mesh)
		boundaries.set_all(0)
		walls.mark(boundaries, 1)
		ds = Measure("ds", domain=mesh, subdomain_data = boundaries)

		f = Expression(("pow(pi,2)*sin(pi*x[1]) - 2*pi*cos(2*pi*x[0])"  ,  "pow(pi,2)*cos(pi*x[0])"))

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

		u_exact = interpolate(u_ex, V_1)
		p_exact = interpolate(p_ex, Q_1)

		n = FacetNormal(mesh)
		t = as_vector((n[1], -n[0]))
		tau_ = 2.*mu*sym(grad(u_))
		tau_ex = 2.*mu*sym(grad(u_exact))
		stress_ = assemble(dot(tau_,n)[0]*ds(1))
		stress_ex = assemble(dot(tau_ex,n)[0]*ds(1))
		sErr = abs(stress_ex - stress_)
		SE.append(sErr)

	Sr = []
	for j in range(1,len(SE)):
		conv = np.log(SE[j-1]/SE[j])/np.log(hvals[j-1]/hvals[j])
		Sr.append(conv)
		print ('h = %f    r = %f' % (hvals[j], conv))
	plt.loglog(hvals,SE, label='P%d-P%d' % ((i[0],i[1])))
plt.legend(loc='upper left')
plt.title('Shear Stress')
plt.savefig('Shear_stress_convergence.png')