# solve problem -laplace=f with homogenous boundary conditions on
# unit interval with analytical solution u = sin(k*pi*x)
# then estimate C for k = 1, 10, 100 in meshes with 100, 1000, 10000 elements

from dolfin import *
import numpy as np
set_log_active(False)

N = [100, 1000, 10000]
K = [1, 10, 100]
print '-------------norms-----------------'
for n in N:
    print '--------------N=%g----------------' %n
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, 'CG', 1)
    V2 = FunctionSpace(mesh, 'CG', 4)
    for k in K:
        print '--------------k=%g---------------' %k

        #u0 = Expression('sin(k*pi*x[0])', k=0, pi=pi)        
    
        def boundary(x, on_boundary):
            return on_boundary

        BC = DirichletBC(V, 0, boundary)
        BC2 = DirichletBC(V2, 0, boundary)

        u = TrialFunction(V)
        v = TestFunction(V)
        f = Expression('k*k*pi*pi*sin(k*pi*x[0])', k=k)
        a = inner(grad(u), grad(v))*dx
        L = f*v*dx

        u_ = Function(V)
        solve(a == L, u_, BC)
    
        #print u.vector().array()[:]
        #r = np.linspace(0, 1, n+1)
        #for i in r:
        #    print np.sin(k*np.pi*i)

        ue = Expression('sin(k*pi*x[0])', k=k)
        uex = project(ue, V2, BC2)
        e = errornorm(uex, u_, 'h1')

        h2norm = np.sqrt(0.5*(1 + (k*np.pi)**2 + (k*np.pi)**4))
        C = e/h2norm*n
        print 'h2norm    = %g' %h2norm
        print 'errornorm = %g' %e
        print 'C         = %g'%C
        print ''
        

        #plot(uex, interactive=True)


