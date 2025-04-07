"""
    TrajectoryCost(dims, idx, stage_cost, terminal_cost)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
struct TrajectoryCost
    dims::PrimalDimensions
    idx::PrimalIndices
    stage_cost::Function
    terminal_cost::Function
end

"""
    cost(yref, y)

Callable struct method for the `TrajectoryCost` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (params::TrajectoryCost)(
    yref::Vector{<:AbstractFloat},
    y::Vector{<:DiffFloat}
)::DiffFloat
    J = 0.0
    for k in 1 : params.dims.N-1
        J += params.stage_cost(
            y[params.idx.x[k]] - yref[params.idx.x[k]],
            y[params.idx.u[k]] - yref[params.idx.u[k]]
        )
    end
    J += params.terminal_cost(
        y[params.idx.x[params.dims.N]] - yref[params.idx.x[params.dims.N]]
    )
    return J
end
