from firedrake import *
Lx = 0.1
Ly = 10.

nx = 5
ny = 100

mesh = RectangleMesh(nx, ny, Lx, Ly)

VX = VectorFunctionSpace(mesh, "CG", 1, dim=3)
V = FunctionSpace(mesh, "CG", 1)
M = MixedFunctionSpace((VX,V))
w = Function(M)

dX, kappa = split(w)

x0 = mesh.coordinates
X0 = Function(VX)
X0.interpolate(as_vector([x0[0], x0[1], 0.]))

X = X0 + dX

nu, chi = TestFunctions(M)

# d/dX = (dX/dx)^{-1}d/dx
J = grad(X)

Jinv = dot(inv(dot(J.T,J)),J.T)
def Xgrad(q):
    return dot(Jinv.T, grad(q).T)

j1 = as_vector([X[0].dx(0),X[1].dx(0),X[2].dx(0)])
j2 = as_vector([X[0].dx(1),X[1].dx(1),X[2].dx(1)])
n_mag = cross(j1,j2)
n = cross(j1,j2)/(dot(n_mag,n_mag)**0.5)

#need to define a normal
#need to include a change of measure

F = (
    inner(kappa*n, nu) + inner(Xgrad(X), Xgrad(nu))
    )*dx
