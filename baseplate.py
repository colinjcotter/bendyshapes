from firedrake import *
Ly = 1.0
Lx = 1.0

nw = 4
nl = 40
deg = 1
mesh = RectangleMesh(nl, nw, Lx, Ly)
X0 = mesh.coordinates

V = VectorFunctionSpace(mesh, "CG", deg, dim=3)
K = FunctionSpace(mesh, "CG", deg)
Rot = Constant(0.1)*2*pi*X0[0]/Lx
Xbcs = Function(V).interpolate(as_vector([X0[0],
                                          cos(Rot)*X0[1],
                                          sin(Rot)*X0[1]]))
X03D = Function(V).interpolate(as_vector([X0[0],X0[1],0.]))
    
Dt = 1.0e-3
dt = Constant(Dt)

Xn = Function(V).assign(Xbcs)
Xnp = Function(V).assign(Xn)
eta = TestFunction(V)

J = grad(Xnp)
Jdag = inv(dot(J.T, J))*J.T
Jcross = cross(Xnp.dx(0),Xnp.dx(1))
detJ = inner(Jcross,Jcross)**0.5

F = (
    inner(Xnp - Xn, eta) + dt*inner(dot(Jdag.T,grad(Xnp).T),
                                         dot(Jdag.T,grad(eta).T)*detJ)
    )*dx

bcs = [DirichletBC(V, Xbcs, (1,2))]
prob = NonlinearVariationalProblem(F, Xnp, bcs=bcs)

solver = NonlinearVariationalSolver(prob,
                                    solver_parameters=
                                    {'mat_type': 'aij',
                                     'snes_converged_reason':True,
                                     'ksp_converged_reason':True,
                                     'snes_linesearch_type':'basic',
                                     "snes_monitor":True,
                                     'ksp_type': 'preonly',
                                     'pc_type': 'lu'})

T = 0.5
t = 0.

file = File('baseplateflow.pvd')

dX = Function(V)

dX.assign(Xnp-X03D)
file.write(dX)

while t < T - Dt/2:
    print(t)
    t += Dt
    solver.solve()

    dX.assign(Xnp-X03D)
    file.write(dX)
