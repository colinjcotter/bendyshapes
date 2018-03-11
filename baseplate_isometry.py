from firedrake import *
Ly = 1.0
Lx = 1.0

nw = 40
nl = 40

mesh = RectangleMesh(nl, nw, Lx, Ly)
X0 = mesh.coordinates
V = VectorFunctionSpace(mesh, "CG", deg=1, dim=3)
X03D = Function(V).interpolate(as_vector([X0[0],X0[1],0.]))

deg = 2
mesh = Mesh(X03D)
X0 = mesh.coordinates
V = VectorFunctionSpace(mesh, "CG", deg)
Q = FunctionSpace(mesh, "CG", deg-1)

Dt = 1.0e-5
dt = Constant(Dt)

Xn = Function(V).assign(X0)

W = FunctionSpace((V,Q))
w = Function(W)
Xnp, P = w.split()
Xnp.assign(Xn)

Xnp, P = split(w)

eta, Sig = TestFunction(W)

def grad2D(Z):
    return as_tensor([[Z[0].dx(0), Z[0].dx(1)],
                      [Z[1].dx(0), Z[1].dx(1)],
                      [Z[2].dx(0), Z[2].dx(1)]])

J = grad2D(Xnp)
Jdag = inv(dot(J.T, J))*J.T
Jcross = cross(Xnp.dx(0),Xnp.dx(1))
detJ = inner(Jcross,Jcross)**0.5

F = (
    inner(Xnp - Xn, eta)*detJ + dt*inner(dot(Jdag.T,grad2D(Xnp).T),
                                         dot(Jdag.T,grad2D(eta).T)*detJ)
    
)*dx

theta= Constant(pi/4)
s = X0[0]/Lx
X_bc0 = as_vector([s*Lx*cos(theta),
                   X0[1],
                   s*Lx*sin(theta)])
X_bc1 = as_vector([s*Lx*cos(theta),
                   X0[1],0.])

tval = Constant(0.)
t0 = Constant(0.25)

bcs = [DirichletBC(W.sub(0), X_bc0 + tval*(X_bc1-X_bc0), (1,2))]
prob = NonlinearVariationalProblem(F, w, bcs=bcs)

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

Xnp, P = w.split()

mesh.coordinates.assign(Xnp)
file.write(Xnp)
mesh.coordinates.assign(X0)

while t < T - Dt/2:
    print(t)
    t += Dt
    tval.assign(max(t/t0, 1.0))
    solver.solve()

    mesh.coordinates.assign(Xnp)
    file.write(Xnp)
    mesh.coordinates.assign(X0)

    Xn.assign(Xnp)
