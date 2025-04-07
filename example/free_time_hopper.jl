using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using HybridRobotDynamics
using HybridTrajIpopt

# Define elast bouncing ball model
system = hopper()

# Define stage and terminal cost functions
Q = 1e0 * I(system.nx)
R = 1e-6 * I(system.nu)
Qf = 1e3 * Q
stage = (x,u) -> x'*Q*x + u'*R*u
terminal = x -> x'*Qf*x

# Define trajopt parameters
N = 25
Δtlb = 5e-3
Δtub = 5e-1
hs = ImplicitIntegrator(:hermite_simpson)
params = ProblemParameters(
    system, hs, stage, terminal, N;
    #Δtlb=Δtlb, Δtub=Δtub
)

# Define transition sequence and terminal guard
impact = system.transitions[:impact]
liftoff = system.transitions[:liftoff]
sequence = [
    TransitionTiming(10, impact),
    TransitionTiming(15, liftoff),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 4.0; 0.0; 5.0; 2.0; 1.0; 1.0; 1.0]
xgc = [5.0; 5.0; 5.0; 5.0; 0.0; 0.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Initial guess
Δt = 0.05
Δts = fill(Δt, N-1)
us = urefs

rk4 = ExplicitIntegrator(:rk4)
xs = roll_out(system, rk4, N, Δt, us, xic, :impact)
plot_2d_states(
    N, system.nx, (1,2), xs;
    title = "Initial Guess",
    xlim = (-10, 10),
    ylim = (-10, 10)
)

# Solve using Ipopt
y0 = compose_trajectory(params.dims, params.idx, xs, us)
cb = SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true
)
sol = ipopt_solve(params, cb, y0)

# Visualize
xs, us, Δts = decompose_trajectory(params.idx, sol.x)
plot_2d_states(N, system.nx, (1,2), xs;)
nothing
