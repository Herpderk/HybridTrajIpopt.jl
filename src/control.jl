const VectorOrMatrix = Union{Vector{<:AbstractFloat}, Matrix{<:AbstractFloat}}

"""
    get_tvlqr_gains(params, integrator, Q, R, Qf, sequence, yref)

Computes the time-varying LQR gains about a nominal trajectory by solving the finite-horizon LQR problem via Riccati recursion.
"""
function get_tvlqr_gains(
    params::ProblemParameters,
    integrator::ExplicitIntegrator,
    Q::Matrix{<:AbstractFloat},
    R::VectorOrMatrix,
    Qf::Matrix{<:AbstractFloat},
    sequence::Vector{TransitionTiming},
    yref::Vector{<:AbstractFloat}
)::Vector{<:VectorOrMatrix}
    # Get problem parameters
    N = params.dims.N
    nx = params.dims.nx
    nu = params.dims.nu
    Δt = params.Δt

    # Init recursion matrices
    Ks = [zeros(nu, nx) for k = 1 : (N-1)]
    Ps = [zeros(nx, nx) for k = 1 : N]
    Ps[N] = 1 * Qf

    # Init transition timing members
    seq_idx = 1
    flow = sequence[seq_idx].transition.flow_I

    # Recursively compute LQR gains
    for k = (params.dims.N-1) : -1 : 1
        # Get the next transition and current flow
        if k == sequence[seq_idx].k
            reset = flow = sequence[seq_idx].transition.reset
            flow = sequence[seq_idx].transition.flow_J
            dynamics = (x, u) -> reset(flow(x, u))
            seq_idx = k == sequence[end].k ? seq_idx : seq_idx + 1
        else
            dynamics = flow
        end

        xref = yref[params.idx.x[k]]
        uref = yref[params.idx.u[k]]
        A = ForwardDiff.jacobian(dx -> integrator(dynamics, dx, uref, Δt), xref)
        B = ForwardDiff.jacobian(du -> integrator(dynamics, xref, du, Δt), uref)

        # Riccati update
        P = Ps[k+1]
        Ks[k] = (R + B'*P*B) \ (B'*P*A)
        Ps[k] = Q + A'*P * (A - B*Ks[k])
    end
    return Ks
end

"""
    TimeVaryingLQR(params, integrator, Q, R, Qf, sequence, yref)

Callable struct that instantiates the time-varying LQR policy gains and references.
"""
struct TimeVaryingLQR
    idx::PrimalIndices
    yref::Vector{<:AbstractFloat}
    gains::Vector{<:VectorOrMatrix}
    function TimeVaryingLQR(
        params::ProblemParameters,
        integrator::ExplicitIntegrator,
        Q::Matrix{<:AbstractFloat},
        R::VectorOrMatrix,
        Qf::Matrix{<:AbstractFloat},
        sequence::Vector{TransitionTiming},
        yref::Vector{<:AbstractFloat}
    )::TimeVaryingLQR
        gains = get_tvlqr_gains(params, integrator, Q, R, Qf, sequence, yref)
        return new(params.idx, yref, gains)
    end
end

"""
    tvlqr(x, k)

Callable struct method for the `TimeVaryingLQR` struct that computes the control input for a given state and time step index.
"""
function (tvlqr::TimeVaryingLQR)(
    x::Vector{<:AbstractFloat},
    k::Int
)::Vector{<:AbstractFloat}
    xref = tvlqr.yref[tvlqr.idx.x[k]]
    uref = tvlqr.yref[tvlqr.idx.u[k]]
    K = tvlqr.gains[k]
    return uref - K * (x - xref)
end

"""
    roll_out(system, integrator, N, Δt, us, x0, init_transition)

Simulates a given system forward in time given an explicit integrator, horizon parameters, control sequence, and initial conditions. Returns the rolled out state trajectory.
"""
function roll_out_tvlqr(
    system::HybridSystem,
    integrator::ExplicitIntegrator,
    N::Int,
    Δt::AbstractFloat,
    tvlqr::TimeVaryingLQR,
    x0::Vector{<:AbstractFloat},
    init_mode::Symbol
)::Vector{<:AbstractFloat}
    # Init loop variables
    xs = [zeros(system.nx) for k = 1:N]
    xs[1] = x0
    mI = system.modes[init_mode]

    # Init timing mapping from roll-out to TVLQR
    kmap = length(tvlqr.idx.u) / N

    # Roll out over time horizon
    for k = 1:N-1
        # Integrate smooth dynamics
        uk = tvlqr(xs[k], trunc(Int, 1 + k*kmap))
        xs[k+1] = integrator(mI.flow, xs[k], uk, Δt)

        # Reset and update mode if guard is hit
        for (trans, mJ) in mI.transitions
            if trans.guard(xs[k+1]) <= 0.0
                xs[k+1] = trans.reset(xs[k+1])
                mI = mJ
                break
            end
        end
    end
    return vcat(xs...)
end
