from firedrake import *
Lx = 0.1
Ly = 10.

ref_level = 3
deg = 2
mesh = UnitIcosahedralSphereMesh(ref_level, degree=deg)
Xn = mesh.coordinates
V = VectorFunctionSpace(mesh, "CG", deg)
K = FunctionSpace(mesh, "CG", deg)
Xs = Function(V)

Xn.assign(Xn*(1.0+Xn[0])**2)

mesh.init_cell_orientations(Xn)
nu = TestFunction(V)

W = MixedFunctionSpace((V,K))
w = Function(W)
Xnp, kappa = split(w)
eta, chi = TestFunctions(W)

nu = CellNormal(mesh)
Dt = 0.01
dt = Constant(Dt)

F = (
    inner(Xnp - Xn, chi*nu) - dt*kappa*chi +
    inner(kappa*nu, eta) + inner(grad(Xnp), grad(eta))
    )*dx

prob = NonlinearVariationalProblem(F, w)
solver = NonlinearVariationalSolver(prob,
                                    solver_parameters=
                                    {'mat_type': 'aij',
                                     'snes_linesearch_type':'basic',
                                     "snes_monitor":True,
                                     'ksp_type': 'preonly',
                                     'pc_type': 'lu'})

T = 10.
t = 0.

file = File('curvatureflow.pvd')

while t < T - Dt/2:
    solver.solve()

    X_out, kappa_out = w.split()
    mesh.coordinates.assign(X_out)
    Xs.assign(X_out)
    file.write(Xs)
