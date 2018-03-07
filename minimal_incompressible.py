from firedrake import *
Lx = 0.1
Ly = 10.

ref_level = 3
deg = 2
#mesh = UnitIcosahedralSphereMesh(ref_level, degree=deg)
mesh = UnitOctahedralSphereMesh(ref_level, degree=deg, hemisphere="north")
Xn = mesh.coordinates
mesh.init_cell_orientations(Xn)
V = VectorFunctionSpace(mesh, "CG", deg)
K = FunctionSpace(mesh, "CG", deg)
Xbcs = Function(V).interpolate(Xn + 0.7*as_vector([0.,0.,Xn[0]*Xn[1]]))
mesh.coordinates.assign(Xbcs)

V = VectorFunctionSpace(mesh, "CG", deg)
Q = FunctionSpace(mesh, "CG", deg-1)
W = MixedFunctionSpace((V,Q))

w = Function(W)
Xnp, p = w.split()

Xnp.assign(Xbcs)
Xoot = Function(V)
Xoot.assign(Xnp-Xn)
poot = Function(Q)
poot.assign(p)

eta, q = TestFunctions(W)

nu = CellNormal(mesh)
Dt = 1.0e-3
dt = Constant(Dt)

Xnp, p = split(w)

J = Jacobian(mesh)
b1 = J[:,0]
b2 = J[:,1]
bp1 = dot(grad(Xnp),b1)
bp2 = dot(grad(Xnp),b2)
nup_unscaled = cross(bp1,bp2)
nup = nup_unscaled/(dot(nup_unscaled,nup_unscaled)**0.5)

P = Identity(3) - outer(nu,nu)

F = (
    inner(Xnp - Xn, eta) + dt*inner(grad(Xnp),grad(eta))
    -dt*inner(div(eta), p)
    +q*(det(dot(grad(Xnp),P) + outer(nup,nu))-1)
    )*dx

prob = NonlinearVariationalProblem(F, w)

solver = NonlinearVariationalSolver(prob,
                                    solver_parameters=
                                    {'mat_type': 'aij',
                                     'snes_converged_reason':True,
                                     'ksp_converged_reason':True,
                                     "snes_monitor":True,
                                     'ksp_type': 'preonly',
                                     'pc_factor_mat_solver_package': 'mumps',
                                     'pc_type': 'lu'})

T = 10.0
t = 0.

tfile = File('incompressiblecurvatureflow.pvd')
tfile.write(Xoot,poot)

while t < T - Dt/2:
    print(t)
    t += Dt
    solver.solve()

    Xnp, p = w.split()
    Xoot.assign(Xnp-Xn)
    poot.assign(p)
    tfile.write(Xoot,poot)
    mesh.coordinates.assign(Xnp)

