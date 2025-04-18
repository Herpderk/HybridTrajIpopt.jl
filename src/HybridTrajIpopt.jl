module HybridTrajIpopt

using LinearAlgebra
using SparseArrays
using ForwardDiff
using Ipopt
using Plots

using HybridRobotDynamics:
        Flow,
        Transition,
        HybridSystem,
        ExplicitIntegrator

export
        ProblemParameters,
        SolverCallbacks,
        ipopt_solve,
        ImplicitIntegrator,
        TransitionTiming,
        compose_trajectory,
        decompose_trajectory,
        TimeVaryingLQR,
        roll_out_tvlqr

include("utils.jl")
include("indexing.jl")
include("integrators.jl")
include("objective.jl")
include("solver.jl")
include("constraints.jl")
include("control.jl")

end # module HybridTrajIpopt
