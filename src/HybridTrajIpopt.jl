module HybridTrajIpopt

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Ipopt
using Plots

using HybridRobotDynamics: Transition, HybridSystem, ExplicitIntegrator

export
        ProblemParameters,
        SolverCallbacks,
        ipopt_solve,
        ImplicitIntegrator,
        TransitionTiming,
        roll_out,
        compose_trajectory,
        decompose_trajectory,
        bouncing_ball,
        hopper,
        TimeVaryingLQR

include("utils.jl")
include("indexing.jl")
include("integrators.jl")
include("objectives.jl")
include("solver.jl")
include("constraints.jl")
include("control.jl")

end # module HybridTrajIpopt
