"""
    ProblemParameters(
        system, integrator, stage_cost, terminal_cost, N;
        Δt=nothing, Δtlb=nothing, Δtub=nothing
    )

Contains the parameters for a trajectory optimization problem with an assumed non-time-varying quadratic objective.
"""
struct ProblemParameters
    system::HybridSystem
    integrator::ImplicitIntegrator
    objective::TrajectoryCost
    dims::PrimalDimensions
    idx::PrimalIndices
    Δt::Union{Nothing, AbstractFloat}
    Δtlb::Union{Nothing, AbstractFloat}
    Δtub::Union{Nothing, AbstractFloat}
    stage_ineq_constr::Union{Nothing, Function}
    stage_eq_constr::Union{Nothing, Function}
    term_ineq_constr::Union{Nothing, Function}
    term_eq_constr::Union{Nothing, Function}
    function ProblemParameters(
        system::HybridSystem,
        integrator::Integrator,
        stage_cost::Function,
        terminal_cost::Function,
        N::Int;
        Δt::Union{Nothing, AbstractFloat} = nothing,
        Δtlb::Union{Nothing, AbstractFloat} = nothing,
        Δtub::Union{Nothing, AbstractFloat} = nothing,
        stage_ineq_constr::Union{Nothing, Function} = nothing,
        stage_eq_constr::Union{Nothing, Function} = nothing,
        term_ineq_constr::Union{Nothing, Function} = nothing,
        term_eq_constr::Union{Nothing, Function} = nothing
    )::ProblemParameters
        if isnothing(Δt)
            nt = 1
            Δtlb = isnothing(Δt) & isnothing(Δtlb) ? 1e-3 : Δtlb
        else
            nt = 0
        end

        dims = PrimalDimensions(N, system.nx, system.nu, nt)
        idx = PrimalIndices(dims)
        obj = TrajectoryCost(dims, idx, stage_cost, terminal_cost)

        if typeof(integrator) == ExplicitIntegrator
            integrator = ImplicitIntegrator(integrator)
        end

        return new(
            system, integrator, obj, dims, idx,
            Δt, Δtlb, Δtub,
            stage_ineq_constr, stage_eq_constr,
            term_ineq_constr, term_eq_constr
        )
    end
end

"""
    assert_timings(params, sequence)

Raises an error if a transition time step is not at least 2 more than the first time step or previous transition time step. Also raises an error if the final transition time step is >= the horizon length N.
"""
function assert_timings(
    params::ProblemParameters,
    sequence::Vector{TransitionTiming}
)::Nothing
    k = 1
    for timing = sequence
        timing.k < k+2 ? error(
            "Please have at least 2 time steps between transitions!") : nothing
        k = timing.k
    end
    sequence[end].k >= params.dims.N ? error(
        "Final transition time step should be < horizon length N!") : nothing
    return
end

"""
    SolverCallbacks(params, sequence, term_guard, xrefs, urefs, xic, xgc)

Contains solver-agnostic callback functions, constraint jacobian and Lagrangian hessian sparsity patterns, and dual variable dimensions.
"""
struct SolverCallbacks
    f::Function
    Lc::Function
    g::Function
    h::Function
    c::Function
    f_grad::Function
    g_jac::Function
    h_jac::Function
    c_jac::Function
    f_hess::Function
    Lc_hess::Function
    c_jac_sp::SparsityPattern
    L_hess_sp::SparsityPattern
    dims::DualDimensions
    gauss_newton::Bool
    function SolverCallbacks(
        params::ProblemParameters,
        sequence::Vector{TransitionTiming},
        term_guard::Function,
        xrefs::Vector{<:AbstractFloat},
        urefs::Vector{<:AbstractFloat},
        xic::Vector{<:AbstractFloat},
        xgc::Union{Nothing, Vector{<:AbstractFloat}} = nothing;
        gauss_newton::Bool = false
    )::SolverCallbacks
        # Enforce transition sequence timing rules
        assert_timings(params, sequence)

        # Define objective
        yref = compose_trajectory(params.dims, params.idx, xrefs, urefs)
        f = y::Vector{<:DiffFloat} -> params.objective(yref, y)

        # Compose inequality constraints
        gs = Vector{Function}([
            y::Vector{<:DiffFloat} -> guard_keepout(params, sequence, term_guard, y);
        ])
        if !isnothing(params.system.stage_ineq_constr)
            push!(gs, y::Vector{<:DiffFloat} -> stage_inequality_constraint(params, y))
        end
        if !isnothing(params.system.term_ineq_constr)
            push!(gs, y::Vector{<:DiffFloat} -> terminal_inequality_constraint(params, y))
        end
        g = y::Vector{<:DiffFloat} -> vcat([g(y) for g = gs]...)

        # Compose equality constraints
        hs = Vector{Function}([
            y::Vector{<:DiffFloat} -> initial_condition(params, xic, y);
            y::Vector{<:DiffFloat} -> dynamics_defect(params, sequence, y, params.Δt);
            y::Vector{<:DiffFloat} -> guard_touchdown(params, sequence, y)
        ])
        if !isnothing(xgc)
            push!(hs, y::Vector{<:DiffFloat} -> goal_condition(params, xgc, y))
        end
        if !isnothing(params.system.stage_eq_constr)
            push!(hs, y::Vector{<:DiffFloat} -> stage_equality_constraint(params, y))
        end
        if !isnothing(params.system.term_eq_constr)
            push!(hs, y::Vector{<:DiffFloat} -> terminal_equality_constraint(params, y))
        end
        h = y::Vector{<:DiffFloat} -> vcat([h(y) for h = hs]...)

        # Compose all constraints
        c = y::Vector{<:DiffFloat} -> [g(y); h(y)]

        # Define constraint component of Lagrangian
        Lc = (y::Vector{<:DiffFloat}, λ::Vector{<:AbstractFloat}) -> λ' * c(y)

        # Get constraint / dual variable dimensions
        yinf = fill(Inf, params.dims.ny)
        ng = length(g(yinf))
        nh = length(h(yinf))
        dims = DualDimensions(ng, nh)

        # Autodiff all callbacks
        println("forward diffing...")
        f_grad = y::Vector{<:DiffFloat} -> ForwardDiff.gradient(f, y)
        g_jac = y::Vector{<:DiffFloat} -> ForwardDiff.jacobian(g, y)
        h_jac = y::Vector{<:DiffFloat} -> ForwardDiff.jacobian(h, y)
        c_jac = y::Vector{<:DiffFloat} -> ForwardDiff.jacobian(c, y)
        f_hess = y::Vector{<:DiffFloat} -> ForwardDiff.hessian(f, y)
        if gauss_newton
            Lc_hess = (
                y::Vector{<:DiffFloat},
                λ::Vector{<:AbstractFloat}
            ) -> zeros(params.dims.ny, params.dims.ny)
        else
            Lc_hess = (
                y::Vector{<:DiffFloat},
                λ::Vector{<:AbstractFloat}
            ) -> ForwardDiff.hessian(dy::Vector{<:DiffFloat} -> Lc(dy, λ), y)
        end

        # Get constraint jacobian and Lagrangian hessian sparsity patterns
        println("getting sparsity patterns...")
        λinf = fill(Inf, dims.nc)
        c_jac_sp = SparsityPattern(c_jac(yinf))
        L_hess_sp = SparsityPattern(f_hess(yinf) + Lc_hess(yinf, λinf))
        return new(
            f, Lc, g, h, c,
            f_grad, g_jac, h_jac, c_jac,
            f_hess, Lc_hess,
            c_jac_sp, L_hess_sp,
            dims, gauss_newton
        )
    end
end

"""
    IpoptCallbacks(cb)

Contains callbacks for the Julia-wrapped Ipop C interface according to the
solver's documentation: https://github.com/jump-dev/Ipopt.jl/tree/master.
"""
struct IpoptCallbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Function
    function IpoptCallbacks(
        cb::SolverCallbacks
    )::IpoptCallbacks
        function eval_f(        # Objective evaluation
            x::Vector{Float64}
        )::Float64
            return cb.f(x)
        end

        function eval_g(        # Constraint evaluation
            x::Vector{Float64},
            g::Vector{Float64}
        )::Nothing
            g .= cb.c(x)
            return
        end

        function eval_grad_f(   # Objective gradient
            x::Vector{Float64},
            grad_f::Vector{Float64}
        )::Nothing
            grad_f .= cb.f_grad(x)
            return
        end

        function eval_jac_g(    # Constraint jacobian
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows .= cb.c_jac_sp.row_coords
                cols .= cb.c_jac_sp.col_coords
            else
                #values .= sparse(cb.c_jac(x)).nzval
                c_jac = cb.c_jac(x)
                @inbounds @simd for i = 1:cb.c_jac_sp.nzvals
                    values[i] = c_jac[
                        cb.c_jac_sp.row_coords[i],
                        cb.c_jac_sp.col_coords[i]
                    ]
                end
            end
            return
        end

        function eval_h(        # Lagrangian hessian
            x::Vector{Float64},
            rows::Vector{Cint},
            cols::Vector{Cint},
            obj_factor::Float64,
            lambda::Vector{Float64},
            values::Union{Nothing, Vector{Float64}}
        )::Nothing
            if isnothing(values)
                rows .= cb.L_hess_sp.row_coords
                cols .= cb.L_hess_sp.col_coords
            else
                L_hess = obj_factor * cb.f_hess(x) + cb.Lc_hess(x, lambda)
                @inbounds @simd for i = 1:cb.L_hess_sp.nzvals
                    values[i] = L_hess[
                        cb.L_hess_sp.row_coords[i],
                        cb.L_hess_sp.col_coords[i]
                    ]
                end
            end
            return
        end
        return new(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    end
end

"""
    ipopt_solve(params, cb, y0, print_level; approx_hessian)

Sets up and solves a trajectory optimization using Ipopt given a set of problem parameters and solver callback functions.
"""
function ipopt_solve(
    params::ProblemParameters,
    cb::SolverCallbacks,
    y0::Vector{<:AbstractFloat};
    print_level::Int = 5,
    max_iter::Int = 1000,
    tol::AbstractFloat = 1e-6
)::IpoptProblem
    # Define primal and constraint bounds
    cb_ipopt = IpoptCallbacks(cb)
    ylb = fill(-Inf, params.dims.ny)
    yub = fill(Inf, params.dims.ny)
    clb = [fill(-Inf, cb.dims.ng); zeros(cb.dims.nh)]
    cub = zeros(cb.dims.nc)

    # Add bounds on time steps
    if !isnothing(params.Δtlb)
        @simd for k = 1 : params.dims.N-1
            ylb[params.idx.Δt[k]] .= params.Δtlb
        end
    end
    if !isnothing(params.Δtub)
        @simd for k = 1 : params.dims.N-1
            yub[params.idx.Δt[k]] .= params.Δtub
        end
    end

    # Create Ipopt problem
    prob = Ipopt.CreateIpoptProblem(
        params.dims.ny,
        ylb,
        yub,
        cb.dims.nc,
        clb,
        cub,
        cb.c_jac_sp.nzvals,
        cb.L_hess_sp.nzvals,
        cb_ipopt.eval_f,
        cb_ipopt.eval_g,
        cb_ipopt.eval_grad_f,
        cb_ipopt.eval_jac_g,
        cb_ipopt.eval_h
    )

    # Add solver options
    Ipopt.AddIpoptIntOption(prob, "print_level", print_level)
    Ipopt.AddIpoptIntOption(prob, "max_iter", max_iter)
    Ipopt.AddIpoptNumOption(prob, "tol", tol)

    # Warm-start and solve
    println("starting ipopt...")
    prob.x = y0
    status = Ipopt.IpoptSolve(prob)
    return prob
end
