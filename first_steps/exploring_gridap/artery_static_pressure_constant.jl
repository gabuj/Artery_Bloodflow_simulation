using Gridap
using GridapGmsh
using GridapGmsh: get_tag_from_name, get_face_labeling, add_tag_from_tags!

model=DiscreteModelFromFile("Artery_meshes/vtu_meshes/C021_light.msh")
labels = get_face_labeling(model)
###################
# create test space for velocity and pressure
# We will use a Lagrangian finite element space of order 2 for velocity
D = 3
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,labels=labels,dirichlet_tags=["walls"])

# We will use a Lagrangian finite element space of order 1 for pressure
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2, dirichlet_tags=["outlet2","outlet1","inlet"]) 
###################
# create trial space for velocity and pressure
#set Dirichlet boundary conditions for velocity (0 at walls)
VD0 = VectorValue(0,0,0)

#set Dirichlet boundary conditions for pressure (10 at inlet and 0 at outlets)
PD0 = 100.0
PD1 = 0.0
PD2 = 0.0

U = TrialFESpace(V,[VD0])
P = TrialFESpace(Q,[PD0,PD1,PD2])


Y = MultiFieldFESpace([V, Q]) #sort of get it but will understand later why put both in the same space
X = MultiFieldFESpace([U, P])

###################
# set up numerical integration for the weak form
degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)


#around inlet and outlets
Γ_i = BoundaryTriangulation(model,tags=["inlet"])
dΓ_i = Measure(Γ_i,degree)
n_Γ_i = get_normal_vector(Γ_i)

Γ_o = BoundaryTriangulation(model,tags=["outlet1","outlet2"])
dΓ_o = Measure(Γ_o,degree)
n_Γ_o = get_normal_vector(Γ_o)

###################
#define weak form functions/terms
const Re = 10.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

###################
#write the weak form
#linear part
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ  #no boundary terms here. Boundary term is: 

#nonlinear part

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

#neumann pressure boundary condition
h_i=-10
h_o=0.0

neumann(v)= -∫(( v·n_Γ_i)*h_i)dΓ_i - ∫(( v·n_Γ_o)*h_o)dΓ_o

#residual and jacobian
res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) - neumann(v)
jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

###############
#setup FE problem
op = FEOperator(res,jac,X,Y)

###############
#set up solver
using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

###############
#solve the problem
uh, ph = solve(solver,op)

#save the solution
outputfile = "first_steps/tutorial_outputs/artery_constant_pressure"
writevtk(Ωₕ,outputfile,cellfields=["uh"=>uh,"ph"=>ph])

println("Solution written to $outputfile")

writevtk(model, "first_steps/tutorial_outputs/artery_constant_pressure_mesh", cellfields=["uh"=>uh,"ph"=>ph])