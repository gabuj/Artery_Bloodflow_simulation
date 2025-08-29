#TUTORIAL 16 AT https://gridap.github.io/Tutorials/dev/pages/t016_poisson_distributed/

using Gridap
using GridapGmsh
using Gridap.MultiField

using LinearAlgebra
using BlockArrays


using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, BiformBlock, BlockTriangularSolver

using GridapDistributed
using PartitionedArrays

using GridapDistributed: writevtk


function main(nparts, distribute)
  parts  = distribute(LinearIndices((nparts,)))
  ###################
  # get model
  model=GmshDiscreteModel(parts, "first_steps/models/cylinder.msh")
  Re = 10.0
  ###################
  # create labelled boundary tags
  # labels = get_face_labeling(model) #this will create a dictionary with the tags of the faces

  ###################
  # create test space for velocity and pressure
  # We will use a Lagrangian finite element space of order 2 for velocity
  D = 3
  order = 2
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  V = TestFESpace(model,reffeᵤ;conformity=:H1,dirichlet_tags=["wall", "outlet", "inlet"])

  # We will use a Lagrangian finite element space of order 1 for pressure
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
  Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)
  ###################
  # create trial space for velocity and pressure
  #set Dirichlet boundary conditions for velocity
  uDwall = VectorValue(0,0,0)
  uDinlet = VectorValue(0,1,0) #this is the velocity at the inlet boundary
  U = TrialFESpace(V,[uDwall,uDwall, uDinlet])
  P = TrialFESpace(Q)

  mfs = BlockMultiFieldStyle(2,(1,1),(1,2))

  Y = MultiFieldFESpace([V, Q]; style=mfs) #sort of get it but will understand later why put both in the same space
  X = MultiFieldFESpace([U, P]; style=mfs)

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
  p_inlet= 1000
  p_out=0
  h_vflux_i= 10
  h_vflux_o= 0

  #pressure neumann boundary conditions with free flux
  neumann(u,v)=  ∫( (v·n_Γ_i) * p_inlet )dΓ_i + ∫( (v·n_Γ_o) * p_out )dΓ_o #- ∫( v·(∇(u)·n_Γ_i))dΓ_i - ∫( v·(∇(u)·n_Γ_o))dΓ_o

  dneumann(du,v)= ∫( v·(∇(du)·n_Γ_i))dΓ_i + ∫( v·(∇(du)·n_Γ_o))dΓ_o


  #residual and jacobian
  res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v) + neumann(u,v)
  jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v) # +dneumann(du,v)

  ###############
  #setup FE problem
  op = FEOperator(res,jac,X,Y)

  solver_u = LUSolver()
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)
  solver_p.log.depth = 4

  α = 1.e2


  u_block = NonlinearSystemBlock()
  p_block = BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)

  bblocks  = [   u_block             LinearSystemBlock();
          LinearSystemBlock()      p_block       ]
  coeffs = [1.0 1.0;
          0.0 1.0]
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-6,rtol=1.e-12,verbose=i_am_main(parts))
  solver.log.depth = 2

  ###############
  #set up solver
  nls = NewtonSolver(solver;maxiter=20,atol=1e-6,rtol=1.e-12,verbose=i_am_main(parts))


  ###############
  #solve the problem
  uh, ph = solve(nls,op)

  #save the solution
 outputfile = "first_steps/tutorial_outputs/exploring_neumann_boundaries/NS_neumann_cylinder_parallel"
 writevtk(Ωₕ,outputfile,cellfields=["uh"=>uh,"ph"=>ph])
 print("Solution saved to ", outputfile)end 
 # run the main function with the number of parts and whether to distribute or not

nparts = 4
with_mpi() do distribute
  main(nparts,distribute)
end