module Explicit
    using ..SaltedDIRCOL: DiffFloat, DiffVector

    """
        rk4(dynamics, x0, u0, x1, Δt)

    Explicit integrator using the fourth order Runge-Kutta method. Written as an equality constraint.
    """
    function rk4(
        dynamics::Function,
        x0::DiffVector,
        u0::DiffVector,
        Δt::DiffFloat
    )::DiffVector
        k1 = dynamics(x0, u0)
        k2 = dynamics(x0 + Δt/2*k1, u0)
        k3 = dynamics(x0 + Δt/2*k2, u0)
        k4 = dynamics(x0 + Δt*k3, u0)
        return x0 + Δt/6*(k1 + 2*k2 + 2*k3 + k4)
    end
end


module Implicit
    using ..SaltedDIRCOL: DiffFloat, DiffVector

    """
        hermite_simpson(dynamics, x0, u0, x1, Δt)

    Implicit integrator using the Hermite-Simpson method.
    """
    function hermite_simpson(
        dynamics::Function,
        x0::DiffVector,
        u0::DiffVector,
        x1::DiffVector,
        Δt::DiffFloat
    )::DiffVector
        f0 = dynamics(x0, u0)
        f1 = dynamics(x1, u0)
        xc = 1/2*(x0 + x1) + Δt/8*(f0 - f1)
        fc = dynamics(xc, u0)
        return x0 - x1 + Δt/6*(f0 + 4*fc + f1)
    end
end


"""
"""
struct ExplicitIntegrator
    method::Function
end
ExplicitIntegrator(method_name::Symbol) = ExplicitIntegrator(
    get_module_function(Explicit, method_name)
)

"""
"""
struct ImplicitIntegrator
    method::Function
end
ImplicitIntegrator(method_name::Symbol) = ImplicitIntegrator(
    get_module_function(Implicit, method_name)
)
ImplicitIntegrator(explicit::ExplicitIntegrator) = ImplicitIntegrator(
    (dynamics, x0, u0, x1, Δt) -> explicit(dynamics, x0, u0, Δt) - x1
)

const Integrator = Union{ExplicitIntegrator, ImplicitIntegrator}

"""
"""
function (integrator::Integrator)(
    dynamics::Function,
    primals::Vararg{Union{DiffVector, DiffFloat}}
)::DiffVector
    return integrator.method(dynamics, primals...)
end
