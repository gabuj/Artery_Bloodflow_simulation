using Gridap

#############
#create model

model = DiscreteModelFromFile("../models/model.json")

#############
#create test and trial spaces
# We will use a Lagrangian finite element space of order 1

#test
order = 1
reffe = ReferenceFE(lagrangian,Float64,order) 
V0 = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="sides")

#trial
g(x) = 2.0
Ug = TrialFESpace(V0,g) #g sets the Dirichlet boundary condition on the trial space

#############
#set up numerial integration for the weak form
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

#neumann boundaries
neumanntags = ["circle", "triangle", "square"]
Γ = BoundaryTriangulation(model,tags=neumanntags)
dΓ = Measure(Γ,degree)

#############
#write the weak form
f(x) = 1.0 
h(x) = 3.0
a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
b(v) = ∫( v*f )*dΩ + ∫( v*h )*dΓ

#############
#setup FE problem
op = AffineFEOperator(a,b,Ug,V0)

#############
#set up solver
ls = LUSolver() 
solver = LinearFESolver(ls)

#############
#solve the problem
uh = solve(solver,op)

#############
#postprocess the solution
#write the solution to a file
outputfile = "first_steps/tutorial_outputs/results"
writevtk(Ω,outputfile,cellfields=["uh"=>uh])
println("Solution written to $outputfile")