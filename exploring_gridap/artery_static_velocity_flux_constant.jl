using Gridap
using GridapGmsh
using GridapGmsh: get_tag_from_name, get_face_labeling

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


Y = MultiFieldFESpace([V, Q]) #sort of get it but will understand later why put both in the same space
X = MultiFieldFESpace([U, P])

###################
# set up numerical integration for the weak form
degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)


#neumann boundaries
#neumanntags = ["outlet2", "outlet1", "inlet"]
Γ_i = BoundaryTriangulation(model,tags=["outlet2"])
dΓ_i = Measure(Γ_i,degree)


####################
#get unit vector normal to cells in Γ_i
n_Γ_cellfield=get_normal_vector(Γ_i)

# # Integrate the normal over Γ_i and divide by the area to get the mean normal vector
# mean_number=sum(∫( n_Γ_cellfield )dΓ_i)
# area = sum(∫( 1.0 )dΓ_i)

# # Mean normal vector
# n_Γ = mean_number / area

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
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ  #to add neumann boundary conditions, add a term like ∫( v*q )dΓ

#boundary condition needs to be added in the residual, if added in a doesn't work... why?!?!?!?! don't know...

#nonlinear part
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

#neumann boundary condition
n(v)= ∫((v⋅n_Γ)*h)dΓ_i

#residual and jacobian
res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) - n(v)
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
outputfile = "first_steps/tutorial_outputs/results"
writevtk(Ωₕ,outputfile,cellfields=["uh"=>uh,"ph"=>ph])

println("Solution written to $outputfile")





# ###############
# #get random initial guess for velocity and pressure
# import Random
# Random.seed!(1234)

# uh_coeffs = rand(Float64,num_free_dofs(U))
# uh0 = FEFunction(U,uh_coeffs)

# ph_coeffs = randn(Float64,num_free_dofs(P))
# ph0= FEFunction(P,ph_coeffs)

# x_coeffs = vcat(uh_coeffs, ph_coeffs)
# initial_guess = FEFunction(X, x_coeffs)

# #solve the problem
# uh, ph = solve!(initial_guess,solver,op)

# #save the solution
# outputfile = "first_steps/tutorial_outputs/results"
# writevtk(Ωₕ,outputfile,cellfields=["uh"=>uh,"ph"=>ph])