from firedrake import *
Lx = 0.1
Ly = 10.

ref_level = 3
deg = 2
#mesh = UnitIcosahedralSphereMesh(ref_level, degree=deg)
mesh = UnitOctahedralSphereMesh(ref_level, degree=deg, hemisphere="north")
Xn = mesh.coordinates
V = VectorFunctionSpace(mesh, "CG", deg)
K = FunctionSpace(mesh, "CG", deg)
Xbcs = Function(V).assign(Xn)

mesh.init_cell_orientations(Xn)
nu = TestFunction(V)

Xnp = Function(V).assign(Xbcs)
eta = TestFunction(V)

nu = CellNormal(mesh)
Dt = 1.0e-5
dt = Constant(Dt)

F = (
    inner(Xnp - Xn, eta) + dt*inner(grad(Xnp),grad(eta))
    )*dx

bcs = [DirichletBC(V, Xbcs, "on_boundary")]
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

file = File('curvatureflow.pvd')

Xnp.assign(Xn)
file.write(Xnp)
while t < T - Dt/2:
    print(t)
    t += Dt
    solver.solve()

    mesh.coordinates.assign(Xnp)
    file.write(Xnp)
