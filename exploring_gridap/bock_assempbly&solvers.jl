#tutorial 24 at https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed/

using LinearAlgebra
using BlockArrays

using Gridap
using Gridap.MultiField

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock
using GridapSolvers.BlockSolvers: BlockDiagonalSolver, BlockTriangularSolver

###############
#geometry
nc = (100,100)
Dc = length(nc)
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1) 

model = CartesianDiscreteModel(domain,nc)
labels = get_face_labeling(model)
if Dc == 2
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
else
  add_tag_from_tags!(labels,"top",[22])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26])
end

#################
#define test and trial spaces
order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

u_walls = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

V = TestFESpace(model,reffe_u,dirichlet_tags=["walls","top"]);
U = TrialFESpace(V,[u_walls,u_top]);
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)

#################
#define blocks 
mfs = BlockMultiFieldStyle(2,(1,1),(1,2))
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

###################
#build the triangulation and measure
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

#define the weak form
α = 1.e1
f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
a((u,p),(v,q)) = ∫(∇(v)⊙∇(u))dΩ - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
l((v,q)) = ∫(v⋅f)dΩ

op = AffineFEOperator(a,l,X,Y)


#####################
#build matrices for linear system
A = get_matrix(op)
b = get_vector(op)


#####################
#build the block system
u_solver = LUSolver()
p_solver = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)

u_block = LinearSystemBlock()
p_block = BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)


#################
#build block preconditioner
####
#block diagonal preconditioner

# PD = BlockDiagonalSolver([u_block,p_block],[u_solver,p_solver])
# solver_PD = FGMRESSolver(20,PD;atol=1e-10,rtol=1.e-12,verbose=true)

# uh, ph = solve(solver_PD, op)

####
#block upper-triangular preconditioner

sblocks = [     u_block             LinearSystemBlock();
           LinearSystemBlock()      p_block       ]
coeffs = [1.0 1.0;
          0.0 1.0]
PU = BlockTriangularSolver(sblocks,[u_solver,p_solver],coeffs,:upper)
solver_PU = FGMRESSolver(20,PU;atol=1e-10,rtol=1.e-12,verbose=true)


uh, ph = solve(solver_PU, op)

#save the solution
outputfile = "first_steps/tutorial_outputs/results_block_upper_triangular"
writevtk(Ω,outputfile,cellfields=["uh"=>uh,"ph"=>ph])