"""
"""
struct ProblemParameters
    integrator::Function
    objective::Function
    dims::PrimalDimensions
    idx::PrimalIndices
    Δt::Union{Nothing, Real}
    function ProblemParameters(
        integrator::Function,
        system::HybridSystem,
        Q::Union{Matrix, Diagonal},
        R::Union{Matrix, Diagonal},
        Qf::Union{Matrix, Diagonal},
        N::Int,
        Δt::Real = nothing
    )::ProblemParameters
        nx = system.nx
        nu = system.nu
        nt = isnothing(Δt) ? 1 : 0
        dims = PrimalDimensions(N, nx, nu, nt)
        idx = PrimalIndices(dims)
        obj = init_quadratic_cost(dims, idx, Q, R, Qf)
        return new(integrator, obj, dims, idx, Δt)
    end
end

"""
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
    function SolverCallbacks(
        params::ProblemParameters,
        sequence::Vector{TransitionTiming},
        term_guard::Function,
        xrefs::Value,
        urefs::Value,
        xic::Value,
        xgc::Union{Nothing, Value} = nothing
    )::SolverCallbacks
        # Define objective
        f = y -> params.objective(xrefs, urefs, y)

        # Compose inequality constraints
        keepout = y -> guard_keepout(params, sequence, term_guard, y)
        g = y -> keepout(y)

        # Compose equality constraints
        ic = y -> initial_condition(params, xic, y)
        defect = y -> dynamics_defect(params, sequence, y, params.Δt)
        touchdown = y -> guard_touchdown(params, sequence, y)
        if isnothing(xgc)
            h = y -> [ic(y); defect(y); touchdown(y)]
        else
            gc = y -> goal_condition(params, xgc, y)
            h = y -> [ic(y); defect(y); touchdown(y); gc(y)]
        end

        # Compose all constraints
        c = y -> [g(y); h(y)]

        # Define constraint component of Lagrangian
        Lc = (y, λ) -> λ' * c(y)

        # Autodiff all callbacks
        f_grad = y -> ForwardDiff.gradient(f, y)
        jacs = [y -> ForwardDiff.jacobian(func, y) for func = (g, h, c)]
        f_hess = y -> ForwardDiff.hessian(f, y)
        Lc_hess = (y, λ) -> ForwardDiff.hessian(dy -> Lc(dy, λ), y)

        # Get constraint / dual variable dimensions
        yinf = fill(Inf, params.dims.ny)
        ng = length(g(yinf))
        nh = length(h(yinf))
        dims = DualDimensions(ng, nh)

        # Get constraint jacobian and Lagrangian hessian sparsity patterns
        λinf = fill(Inf, dims.nc)
        c_jac = jacs[end]
        c_jac_sp = SparsityPattern(c_jac(yinf))
        L_hess_sp = SparsityPattern(f_hess(yinf) + Lc_hess(yinf, λinf))
        return new(
            f, Lc, g, h, c,
            f_grad, jacs...,
            f_hess, Lc_hess,
            c_jac_sp, L_hess_sp,
            dims
        )
    end
end

"""
Documentation: https://github.com/jump-dev/Ipopt.jl/tree/master
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
            x::Value
        )::Float64
            return Float64(cb.f(x))
        end

        function eval_g(        # Constraint evaluation
            x::Value,
            g::Value
        )::Nothing
            g .= cb.c(x)
            return
        end

        function eval_grad_f(   # Objective gradient
            x::Value,
            grad_f::Value
        )::Nothing
            grad_f .= cb.f_grad(x)
            return
        end

        function eval_jac_g(    # Constraint jacobian
            x::Value,
            rows::Vector{Cint},
            cols::Vector{Cint},
            values::Union{Nothing, Value}
        )::Nothing
            if isnothing(values)
                rows .= cb.c_jac_sp.row_idx
                cols .= cb.c_jac_sp.col_idx
            else
                #values .= sparse(cb.c_jac(x)).nzval
                c_jac = cb.c_jac(x)
                @inbounds @simd for i = 1:cb.c_jac_sp.nzvals
                    values[i] = c_jac[
                        cb.c_jac_sp.row_idx[i],
                        cb.c_jac_sp.col_idx[i]
                    ]
                end
            end
            return
        end

        function eval_h(        # Lagrangian hessian
            x::Value,
            rows::Vector{Cint},
            cols::Vector{Cint},
            obj_factor::Real,
            lambda::Value,
            values::Union{Nothing, Value}
        )::Nothing
            if isnothing(values)
                rows .= cb.L_hess_sp.row_idx
                cols .= cb.L_hess_sp.col_idx
            else
                L_hess = obj_factor * cb.f_hess(x) + cb.Lc_hess(x, lambda)
                @inbounds @simd for i = 1:cb.L_hess_sp.nzvals
                    values[i] = L_hess[
                        cb.L_hess_sp.row_idx[i],
                        cb.L_hess_sp.col_idx[i]
                    ]
                end
            end
            return
        end
        return new(eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    end
end

"""
"""
function ipopt_solve(
    params::ProblemParameters,
    cb::SolverCallbacks,
    y0::Value,
    print_level::Int = 5;
    approx_hessian::Bool = false
)::IpoptProblem
    # Define primal and constraint bounds
    cb_ipopt = IpoptCallbacks(cb)
    ylb = fill(-Inf, params.dims.ny)
    yub = fill(Inf, params.dims.ny)
    clb = [fill(-Inf, cb.dims.ng); zeros(cb.dims.nh)]
    cub = zeros(cb.dims.nc)

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
    if approx_hessian
        Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
    end
    Ipopt.AddIpoptIntOption(prob, "print_level", print_level)

    # Warm-start and solve
    prob.x = y0
    solvestat = Ipopt.IpoptSolve(prob)
    return prob
end
