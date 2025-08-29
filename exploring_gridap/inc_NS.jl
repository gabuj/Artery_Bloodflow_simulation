using Gridap

###################
# Create a Cartesian discrete model
n = 100 # Number of divisions in each direction
domain = (0,1,0,1) #(x_min, x_max, y_min, y_max)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

#write the model to a file
# writevtk(model,"2D_square")


###################
# create labelled boundary tags
labels = get_face_labeling(model) #this will create a dictionary with the tags of the faces
add_tag_from_tags!(labels,"diri1",[6,]) #change the tag of the face with tag 6 to "diri1"
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8]) #change the tag of the faces with tags 1,2,3,4,5,7,8 to "diri0"

###################
# create test space for velocity and pressure
# We will use a Lagrangian finite element space of order 2 for velocity
D = 2
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,labels=labels,dirichlet_tags=["diri0", "diri1"])

# We will use a Lagrangian finite element space of order 1 for pressure
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)

###################
# create trial space for velocity and pressure
#set Dirichlet boundary conditions for velocity
uD0 = VectorValue(0,0)
uD1 = VectorValue(1,0)
U = TrialFESpace(V,[uD0, uD1])
P = TrialFESpace(Q)


Y = MultiFieldFESpace([V, Q]) #sort of get it but will understand later why put both in the same space
X = MultiFieldFESpace([U, P])

###################
# set up numerical integration for the weak form
degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

#around inlet and outlets
Γ_i = BoundaryTriangulation(model,tags=["diri1"])
dΓ_i = Measure(Γ_i,degree)
# n_Γ_i = get_normal_vector(Γ_i)
n_Γ_i(x)=VectorValue(0,1)



###################
#define weak form functions/terms
const Re = 10.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

###################
#write the weak form
#linear part
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ

#nonlinear part
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

#neumann pressure boundary condition
h_pi=-10
h_po=0.0
h_vflux= -10

#neumann(v)= -∫(( v·n_Γ_i)*h_pi)dΓ_i - ∫(( v·n_Γ_o)*h_po)dΓ_o
neumann(v)= ∫(( v·n_Γ_i)*h_vflux)dΓ_i


#residual and jacobian
res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) #- neumann(v)
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
outputfile = "first_steps/tutorial_outputs/results_NS"
writevtk(Ωₕ,outputfile,cellfields=["uh"=>uh,"ph"=>ph])