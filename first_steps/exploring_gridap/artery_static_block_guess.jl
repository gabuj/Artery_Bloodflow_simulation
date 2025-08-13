using LinearAlgebra
using BlockArrays

using Gridap
using Gridap.MultiField

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock
using GridapSolvers.BlockSolvers: BlockDiagonalSolver, BlockTriangularSolver
using GridapGmsh
using GridapGmsh: get_tag_from_name, get_face_labeling, add_tag_from_tags!

###############
model=DiscreteModelFromFile("Artery_meshes/vtu_meshes/C021_light.msh")
labels = get_face_labeling(model)
#writevtk(model,"Artery_meshes/gridap_outputs/C021_light")

###################
# create test space for velocity and pressure
# We will use a Lagrangian finite element space of order 2 for velocity
D = 2
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,labels=labels,dirichlet_tags=["walls"]) #flux at inlet is constant

# We will use a Lagrangian finite element space of order 1 for pressure
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2, constraint=:zeromean) #if neumann conditions or no conditions: put constraint=:zeromean

###################
# create trial space for velocity and pressure
#set Dirichlet boundary conditions for velocity (0 at walls)
VD0 = VectorValue(0,0,0)

U = TrialFESpace(V,[VD0])
P = TrialFESpace(Q)

#################
#define blocks 
mfs = BlockMultiFieldStyle(2,(1,1),(1,2))
X = MultiFieldFESpace([U,P];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

###################
#build the triangulation and measure
degree = order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

#define the weak form


###################
# set up numerical integration for the weak form

#neumann boundaries
#neumanntags = ["outlet2", "outlet1", "inlet"]
Γ_i = BoundaryTriangulation(model,tags=["outlet2"])
dΓ_i = Measure(Γ_i,degree)


###################
#define weak form functions/terms
const Re = 10.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

###################
#write the weak form

#neumann part:
n_Γ = get_normal_vector(Γ_i)
h = -3.0

#linear part
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ  

#nonlinear part
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

#neumann boundary condition
n(v)= ∫((v⋅n_Γ)*h)dΓ_i
#f= VectorValue(0,0,0)

l((v,q)) = n(v)

linear_op = AffineFEOperator(a,l,X,Y)


#####################
#build matrices for linear system
A = get_matrix(linear_op)
b = get_vector(linear_op)


#####################
#build the block system
α = 1.e1

u_solver = LUSolver()
p_solver = CGSolver(JacobiLinearSolver();atol=1e-14,rtol=1.e-6)

u_block = LinearSystemBlock()
p_block = BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)


#################
#block upper-triangular preconditioner

sblocks = [     u_block        LinearSystemBlock();
           LinearSystemBlock()      p_block       ]
coeffs = [1.0 1.0;
          0.0 1.0]
PU = BlockTriangularSolver(sblocks,[u_solver,p_solver],coeffs,:upper)
solver_PU = FGMRESSolver(20,PU;atol=1e-10,rtol=1.e-12,verbose=true)

initial_guess = solve(solver_PU, linear_op)

# #save the solution
# uh_i, ph_i= initial_guess
# outputfile = "first_steps/tutorial_outputs/linear_artery_block"
# writevtk(Ω,outputfile,cellfields=["uh"=>uh_i,"ph"=>ph_i])



###################
#solve the non linear problem
#residual and jacobian
res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) - n(v)
jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

# Wrap your block linear solver

using LineSearches: BackTracking

nls = NLSolver(
  solver_PU,
  method = :newton,
  show_trace = true,
  linesearch = BackTracking(),
)
###############
#setup FE problem
op = FEOperator(res,jac,X,Y)

###############
#set up solver
solver = FESolver(nls)
###############
#solve the problem
uh, ph = solve!(initial_guess,solver,op)

#save the solution
outputfile = "first_steps/tutorial_outputs/artery_nonlinear_block"
writevtk(Ω,outputfile,cellfields=["uh"=>uf,"ph"=>pf])

println("Solution written to $outputfile")


