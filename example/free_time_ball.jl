using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using HybridRobotDynamics
using HybridTrajIpopt

# Define elast bouncing ball model
system = bouncing_ball()

# Define stage and terminal cost functions
Q = 1e-3 * diagm([1.0, 1.0, 0.0, 0.0])
R = 1e-3 * I(system.nu)
Qf = 1e6 * Q
stage = (x,u) -> x'*Q*x + u'*R*u
terminal = x -> x'*Qf*x

# Define trajopt parameters
N = 40
Δtlb = 5e-3
Δtub = 5e-1
hs = ImplicitIntegrator(:hermite_simpson)
params = ProblemParameters(
    system, hs, stage, terminal, N;
    Δtlb=Δtlb, Δtub=Δtub
)

# Define transition sequence and terminal guard
impact = system.transitions[:impact]
sequence = [
    TransitionTiming(10, impact),
    TransitionTiming(30, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 5.0; 1.0; 0.0]
xgc = [5.0; 5.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callbacks
cb = SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true
)

# Initial guess
Δt = 0.05
us = urefs
rk4 = ExplicitIntegrator(:rk4)
xs = roll_out(system, rk4, N, Δt, us, xic, :downwards)
plot_2d_states(N, system.nx, (1,2), xs; title="Initial Guess")

# Solve
Δts = fill(Δt, N-1)
y0 = compose_trajectory(params.dims, params.idx, xs, us, Δts)
sol = ipopt_solve(params, cb, y0; max_iter=1000)

# Visualize
xs, us, Δts = decompose_trajectory(params.idx, sol.x)
plot_2d_states(N, system.nx, (1,2), xs)
nothing
