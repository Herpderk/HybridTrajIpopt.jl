"""
    get_primals(params, k, y, Δt)

Returns a tuple of (decision) variables for computing dynamics defect residuals for a given time step.
"""
function get_primals(
    params::ProblemParameters,
    k::Int,
    y::Vector{<:DiffFloat},
    Δt::Union{Nothing, AbstractFloat}
)::Tuple{Vector{<:DiffFloat}, Vector{<:DiffFloat}, Vector{<:DiffFloat}, DiffFloat}
    x0 = y[params.idx.x[k]]
    u0 = y[params.idx.u[k]]
    x1 = y[params.idx.x[k+1]]
    Δt = isnothing(Δt) ? y[params.idx.Δt[k]][1] : Δt
    return x0, u0, x1, Δt
end

"""
    dynamics_defect(params, integrator, sequence, y, Δt=nothing)

Computes dynamics defect residuals for a given integration scheme and transition sequence.
"""
function dynamics_defect(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming},
    y::Vector{<:DiffFloat},
    Δt::Union{Nothing, AbstractFloat} = nothing
)::Vector{<:DiffFloat}
    # Init defect residuals and time step counter
    c = [zeros(eltype(y), params.dims.nx) for k = 1:params.dims.N-1]
    k_start = 1

    # Iterate over each hybrid mode and time step
    for timing = sequence
        # Integrate smooth dynamics
        flow_I = timing.transition.flow_I
        for k = k_start : timing.k-1
            c[k] = params.integrator(flow_I, get_primals(params, k, y, Δt)...)
        end

        # Integrate transition dynamics
        flow_J = timing.transition.flow_J
        x0, u0, x1, Δtval = get_primals(params, timing.k, y, Δt)
        xJ = timing.transition.reset(x0)
        c[timing.k] = params.integrator(flow_J, xJ, u0, x1, Δtval)

        # Update starting time step of next mode
        k_start = timing.k + 1
    end

    # Integrate over remaining time steps
    if k_start < params.dims.N
        flow_J = sequence[end].transition.flow_J
        for k = k_start : params.dims.N-1
            c[k] = params.integrator(flow_J, get_primals(params, k, y, Δt)...)
        end
    end
    return vcat(c...)
end

"""
    guard_keepout(params, term_guard, sequence, y)

Computes guard "keep-out" residuals at every time step without touchdown. Requires an additional terminal guard for keep-out after the last transition.
"""
function guard_keepout(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming},
    term_guard::Function,
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    # Init keepout residuals
    c = zeros(eltype(y), params.dims.N - length(sequence))
    k_start = 1
    i = 1

    # Iterate over each hybrid mode and time step
    for timing = sequence
        for k = k_start : timing.k-1
            # Flip the guard to adhere to NLP convention: g(x) <= 0
            c[i] = -timing.transition.guard(y[params.idx.x[k]])
            i += 1
        end
        # Update starting time step; skip time steps with touchdown constraint
        k_start = timing.k + 1
    end

    # Evaluate terminal guard residuals over remaining time steps
    if k_start < params.dims.N
        for k = k_start : params.dims.N
            c[i] = -term_guard(y[params.idx.x[k]])
            i += 1
        end
    end
    return c
end

"""
    guard_touchdown(params, sequence, y)

Computes guard "touchdown" residuals at time steps right before transitions.
"""
function guard_touchdown(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming},
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    c = zeros(eltype(y), length(sequence))
    for (i, timing) in enumerate(sequence)
        c[i] = timing.transition.guard(y[params.idx.x[timing.k]])
    end
    return c
end

"""
    initial_condition(params, xic, y)

Computes initial condition residuals.
"""
function initial_condition(
    params::ProblemParameters,
    xic::Vector{<:AbstractFloat},
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    return y[params.idx.x[1]] - xic
end

"""
    goal_condition(params, xg, y)

Computes goal condition residuals.
"""
function goal_condition(
    params::ProblemParameters,
    xg::Vector{<:AbstractFloat},
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    return y[params.idx.x[params.dims.N]] - xg
end

"""
    stage_inequality_constraint(params, y)

Computes stage inequality constraint residuals.
"""
function stage_inequality_constraint(
    params::ProblemParameters,
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    c = [zeros(eltype(y), params.system.stage_ng) for k = 1 : params.dims.N-1]
    for k = 1 : params.dims.N-1
        xk = y[params.idx.x[k]]
        uk = y[params.idx.u[k]]
        c[k] = params.system.stage_ineq_constr(xk, uk)
    end
    return vcat(c...)
end

"""
    stage_equality_constraint(params, y)

Computes stage equality constraint residuals.
"""
function stage_equality_constraint(
    params::ProblemParameters,
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    c = [zeros(eltype(y), params.system.stage_nh) for k = 1 : params.dims.N-1]
    for k = 1 : params.dims.N-1
        xk = params.idx.x[k]
        uk = params.idx.u[k]
        c[k] = params.system.stage_eq_constr(xk, uk)
    end
    return vcat(c...)
end

"""
    terminal_inequality_constraint(params, y)

Computes terminal inequality constraint residuals.
"""
function terminal_inequality_constraint(
    params::ProblemParameters,
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    return params.system.term_ineq_constr(y[params.idx.x[params.dims.N]])
end

"""
    terminal_equality_constraint(params, y)

Computes terminal equality constraint residuals.
"""
function terminal_equality_constraint(
    params::ProblemParameters,
    y::Vector{<:DiffFloat}
)::Vector{<:DiffFloat}
    return params.system.term_eq_constr(y[params.idx.x[params.dims.N]])
end
