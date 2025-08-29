using Gridap
using Gridap.MultiField
using GridapGmsh

#############
#create model

###################
# # Create a Cartesian discrete model
# n = 100 # Number of divisions in each direction
# m=3
# domain = (0,1,0,m) #(x_min, x_max, y_min, y_max)
# partition = (n,m*n)
# model = CartesianDiscreteModel(domain,partition)

# # #write the model to a file
# #writevtk(model,"2D_square")


# ##################
# # create labelled boundary tags
# labels = get_face_labeling(model) #this will create a dictionary with the tags of the faces
# add_tag_from_tags!(labels,"inlet",[6,]) #change the tag of the face with tag 6 to "diri1"
# add_tag_from_tags!(labels,"wall",[1,2,3,4,7,8]) #change the tag of the faces with tags 1,2,3,4,5,7,8 to "diri0"
# add_tag_from_tags!(labels,"outlet",[5,]) #change the tag of the face with tag 6 to "diri1"












#discrete model
model=GmshDiscreteModel("first_steps/models/cylinder_light.msh")

#############
#create test and trial spaces
# We will use a Lagrangian finite element space of order 1

D = 3


order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["wall"])#flux at inlet is constant

# We will use a Lagrangian finite element space of order 1 for pressure
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2) #if neumann conditions or no conditions: put constraint=:zeromean

###################
# create trial space for velocity and pressure
#set Dirichlet boundary conditions for velocity
uDwalls = (D == 2) ? VectorValue(0,0) : VectorValue(0,0,0)
uDtop = (D == 2) ? VectorValue(0,1) : VectorValue(0,0,10) #this is the velocity at the top boundary
uDbottom = (D == 2) ? VectorValue(0,0) : VectorValue(0,0,10) #this is the velocity at the bottom boundary

U = TrialFESpace(V,[uDwalls])
P = TrialFESpace(Q)

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
n_Γ_i = -get_normal_vector(Γ_i)

Γ_o = BoundaryTriangulation(model,tags=["outlet"])
dΓ_o = Measure(Γ_o,degree)
n_Γ_o = -get_normal_vector(Γ_o)

###################


#############
#write the weak form

a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ 


#neumann pressure boundary condition
p_inlet= 10
p_out=0

#pressure neumann boundary conditions with free flux
b((v,q))=  ∫( (v·n_Γ_i) * p_inlet )*dΓ_i + ∫( (v·n_Γ_o) * p_out )*dΓ_o #- ∫( v·(∇(u)·n_Γ_i))dΓ_i - ∫( v·(∇(u)·n_Γ_o))dΓ_o


################

op = AffineFEOperator(a,b,X,Y)

#############
#set up solver
ls = LUSolver() 
solver = LinearFESolver(ls)

#############
#solve the problem
xh = solve(solver,op)
uh, ph = xh
#############

#write the solution to a file
#save the solution
# Evaluate each field separately so it's no longer a MultiFieldCellField
# Write to VTK
outputfile = "first_steps/tutorial_outputs/exploring_neumann_boundaries/linear_cylinder_fluidthrough"
writevtk(Ωₕ, outputfile,
    cellfields = [
        "uh" => uh,
        "ph" => ph
    ]
)

println("Solution saved to ", outputfile)
