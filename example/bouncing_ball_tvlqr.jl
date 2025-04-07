using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using Revise
using HybridRobotDynamics
using HybridTrajIpopt

# Define elast bouncing ball model
system = bouncing_ball()

# Define stage and terminal cost functions
Q = 1e0 * diagm([1.0, 1.0, 0.0, 0.0])
R = Matrix(1e-6 * I(system.nu))
Qf = 1e3 * Q
stage = (x,u) -> x'*Q*x + u'*R*u
terminal = x -> x'*Qf*x

# Define trajopt parameters
N = 50
Δt = 0.01
hs = ImplicitIntegrator(:hermite_simpson)
params = ProblemParameters(system, hs, stage, terminal, N; Δt)

# Define transition sequence and terminal guard
impact = system.transitions[:impact]
sequence = [
    TransitionTiming(10, impact),
    TransitionTiming(20, impact),
    TransitionTiming(30, impact),
    TransitionTiming(40, impact),
]
term_guard = impact.guard

# Define reference trajectory and initial conditions
xic = [0.0; 10.0; 10.0; 0.0]
xgc = [5.0; 5.0; 0.0; 0.0]
xrefs = repeat(xgc, N)
urefs = zeros((N-1) * system.nu)

# Define solver callbacks
cb = SolverCallbacks(
    params, sequence, term_guard, xrefs, urefs, xic;
    gauss_newton=true
)

# Solve using Ipopt
y0 = zeros(params.dims.ny)
sol = ipopt_solve(params, cb, y0)

# Init TVLQR policy with RK4 integration
yref = sol.x
rk4 = ExplicitIntegrator(:rk4)
tvlqr = TimeVaryingLQR(params, rk4, Q, R, Qf, sequence, yref)

# Simulate system forward in time with TVLQR policy and smaller time steps
speedup = 10
N_sim = N * speedup
Δt_sim = Δt / speedup
xs_sim = roll_out_tvlqr(system, rk4, N_sim, Δt_sim, tvlqr, xic, :downwards)
plot_2d_states(N_sim, system.nx, (1,2), xs_sim)
nothing
