#TO BE FIXED!!!

using Gridap
using LineSearches: BackTracking

using LinearAlgebra
using BlockArrays

using Gridap.MultiField

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, BiformBlock, BlockTriangularSolver
using GridapGmsh

# model=DiscreteModelFromFile("first_steps/Artery_meshes/vtu_meshes/C021_light.msh")
# labels = get_face_labeling(model)

#discrete model

###################
# Create a Cartesian discrete model
n = 100 # Number of divisions in each direction
l=3
domain = (0,1,0,l) #(x_min, x_max, y_min, y_max)
partition = (n,l*n)
model = CartesianDiscreteModel(domain,partition)

# #write the model to a file
#writevtk(model,"2D_square")


##################
# create labelled boundary tags
labels = get_face_labeling(model) #this will create a dictionary with the tags of the faces
add_tag_from_tags!(labels,"inlet",[6,]) #change the tag of the face with tag 6 to "diri1"
add_tag_from_tags!(labels,"wall",[1,2,3,4,7,8]) #change the tag of the faces with tags 1,2,3,4,5,7,8 to "diri0"
add_tag_from_tags!(labels,"outlet",[5,]) #change the tag of the face with tag 6 to "diri1"



# model=GmshDiscreteModel("models/cylinder_lighter.msh")

###################
# create test space for velocity and pressure
# We will use a Lagrangian finite element space of order 2 for velocity
D = 2

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["wall", "inlet"])#flux at inlet is constant

# We will use a Lagrangian finite element space of order 1 for pressure
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2) #if neumann conditions or no conditions: put constraint=:zeromean

###################
# create trial space for velocity and pressure
#set Dirichlet boundary conditions for velocity
# Dirichlet time-dependent velocity on walls (no-slip -> zero)
u_walls(t) = (D == 2) ? x -> VectorValue(0.0,0.0) : x -> VectorValue(0.0,0.0,0.0) # time-function returning field (zero here)
u_inlet(t) = (D == 2) ? x -> VectorValue(0.0,1.0) : x -> VectorValue(0.0,0.0,10.0) # time-function returning field (constant here)
# u_bottom(t) = (D == 2) ? x -> VectorValue(0
U = TransientTrialFESpace(V, [u_walls, u_inlet])    # time-dependent trial space for velocity
P = TransientTrialFESpace(Q)

mfs = BlockMultiFieldStyle(2,(1,1),(1,2))
Y = MultiFieldFESpace([V, Q]; style=mfs) #sort of get it but will understand later why put both in the same space
X = TransientMultiFieldFESpace([U, P]; style=mfs)



###################
# set up numerical integration for the weak form
degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)


#around inlet and outlets
Γ_i = BoundaryTriangulation(model,tags=["inlet"])
dΓ_i = Measure(Γ_i,degree)
n_Γ_i = -get_normal_vector(Γ_i)

Γ_o = BoundaryTriangulation(model,tags=["outlet"])
dΓ_o = Measure(Γ_o,degree)
n_Γ_o = -get_normal_vector(Γ_o)

###################
#define weak form functions/terms
const Re = 1.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

###################
#write the weak form
#linear part
m(t,u, v) = ∫(v ⋅ ∂t(u))dΩ
a(t,(u,p),(v,q)) = ∫(∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ

#nonlinear part
c(t,u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(t,u,du,v) = ∫(v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ


#neumann pressure boundary condition
p_out=0
h_vflux_i= 10
h_vflux_o= 0

#pressure neumann boundary conditions with free flux
# pressure amplitude function
p_inlet_func(t) = 1000.0 * exp(t/10)   # your original


# time-independent (for this time step) neumann functional using pval
neumann_t(t,(u,v)) = ∫((v·n_Γ_i) * p_inlet_func(t) )dΓ_i + ∫( (v·n_Γ_o) * p_out )dΓ_o
#dneumann(du,v)= ∫( v·(∇(du)·n_Γ_i))dΓ_i + ∫( v·(∇(du)·n_Γ_o))dΓ_o


# Residual and jac for this time-step
res(t,(u,p),(v,q)) = m(t,u, v) + a(t,(u,p),(v,q)) + c(t,u,v) - neumann_t(t,(u,v))
jac(t,(u,p),(du,dp),(v,q)) = a(t,(du,dp),(v,q)) + dc(t,u,du,v)
jac_t(t, u, dtu, v) = ∫(v ⋅ dtu)dΩ


###############
#setup FE problem
op = TransientFEOperator(res,jac, jac_t,X,Y)

solver_u = LUSolver()
#iterative solver actually is slower than direct
# solver_u = FGMRESSolver(50, JacobiLinearSolver(); rtol=1e-4, atol=1e-5) 
solver_p = CGSolver(JacobiLinearSolver();maxiter=50,atol=1.e-6,rtol=1.e-12)

α = 1.e2


u_block = NonlinearSystemBlock()
p_block = BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)

bblocks  = [     u_block             LinearSystemBlock();
LinearSystemBlock()      p_block       ]
coeffs = [1.0 1.0;
0.0 1.0]
Block_solver = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(45,Block_solver;atol=8.e-8,rtol=1.e-12,verbose=true)

###############
#set up solver
nls = NewtonSolver(solver;maxiter=10,atol=8.e-5,rtol=1.e-12, verbose=true)


###############
#set up time steps

Δt = 0.05
t0 = 0.0
tF = 1.0
θ = 0.5

#set up transient solver
transient_solver = ThetaMethod(nls, Δt, θ)
# transient_solver= BackwardEuler(nls, Δt)

##############
#solve the problem
# Build initial conditions:
uh0= (D == 2) ? VectorValue(0, 0) : VectorValue(0, 0, 0)
ph0 = 0.0
ic = [uh0, ph0]   # initial condition as a vector of functions
X₀ = interpolate_everywhere(ic,X(t0)) # needs to match the DOF of the space

Xₕₜ= solve(transient_solver, op, t0, tF, X₀)

#save the solution
outputfile = "tutorial_outputs/transient/nonlinear/transient_2D_NS"

# times = 0:Δt:tF

createpvd(outputfile) do pvd
  for (tn,Xₕ) in Xₕₜ
    pvd[tn] = createvtk(Ωₕ, outputfile * "_$tn" * ".vtu",cellfields=["uh"=>Xₕ[1],"ph"=>Xₕ[2]])
    println("Saving time step ", tn)

end
end


println("Solution saved to ", outputfile)   
