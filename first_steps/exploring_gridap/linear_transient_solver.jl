#TUTORIAL 17 AT https://gridap.github.io/Tutorials/dev/pages/t018_transient_nonlinear/


using Gridap
domain = (-1, +1, -1, +1)
partition = (20, 20)
model = CartesianDiscreteModel(domain, partition)

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)

V0 = TestFESpace(model, reffe, dirichlet_tags="boundary")

g(t) = x -> exp(-2 * t)
Ug = TransientTrialFESpace(V0, g)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

α(t) = x -> 1 + (x[1]^2 + x[2]^2) / 4
f(t) = x -> sin(t) * sinpi(x[1]) * sinpi(x[2])

m(t, dtu, v) = ∫(v * dtu)dΩ
a(t, u, v) = ∫(α(t) * ∇(v) ⋅ ∇(u))dΩ
l(t, v) = ∫(v * f(t))dΩ
op = TransientLinearFEOperator((a, m), l, Ug, V0, constant_forms=(false, true))


ls = LUSolver()
Δt = 0.05
θ = 0.5
solver = ThetaMethod(ls, Δt, θ)

#could build a different solver

t0, tF = 0.0, 10.0
uh0 = interpolate_everywhere(g(t0), Ug(t0))
uh = solve(solver, op, t0, tF, uh0)


#save the solution
outputfile = "first_steps/tutorial_outputs/transient/linear/results_linear_transient_solver_esp"

createpvd(outputfile) do pvd
  pvd[0] = createvtk(Ω, outputfile * "_0" * ".vtu", cellfields=["u" => uh0])
  for (tn, uhn) in uh
    pvd[tn] = createvtk(Ω, outputfile * "_$tn" * ".vtu", cellfields=["u" => uhn])
  end
end

println("Solution saved to ", outputfile)   
